"""Shadow-safe official observation/action bridge for the Phase 6G audit.

The bridge maps one native, agent-visible selection observation to a complete
set of legal ordered selections.  It does not claim that this one-step boundary
is a faithful full-game abstraction, and it cannot consume the three-card neural
blueprint.  In live audits it wraps the preserved baseline, round-trips the
baseline's selection, and never changes the action sent to the official engine.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import permutations
import json
from math import perm
from time import perf_counter
from typing import Any, Mapping, Sequence

from .competition import SelectionContract
from .game_theory import (
    AbstractAction,
    ActionDistribution,
    GameBoundaryKind,
    GameBoundaryProvenance,
    InformationSet,
    InformationSetActor,
    OpponentRange,
    PublicBeliefState,
    RangeHypothesis,
)


OFFICIAL_BRIDGE_SCHEMA_VERSION = 1
HIDDEN_CANARY_CARD_ID = 987_654_321


OPTION_FIELDS = frozenset(
    {
        "type",
        "number",
        "area",
        "index",
        "playerIndex",
        "toolIndex",
        "energyIndex",
        "count",
        "inPlayArea",
        "inPlayIndex",
        "attackId",
        "cardId",
        "serial",
        "specialConditionType",
    }
)
OPTION_SEMANTIC_FIELDS = tuple(sorted(OPTION_FIELDS - {"serial"}))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _card_id(card: Any) -> int | None:
    if not isinstance(card, Mapping):
        return None
    value = card.get("id", card.get("cardId"))
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _card_ids(cards: Any) -> list[int | None]:
    if not isinstance(cards, Sequence) or isinstance(cards, (str, bytes, bytearray)):
        return []
    return [_card_id(card) for card in cards]


def _pokemon_view(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "id": _card_id(value),
        "hp": _integer(value.get("hp")),
        "max_hp": _integer(value.get("maxHp")),
        "appear_this_turn": bool(value.get("appearThisTurn", False)),
        "energies": [
            _integer(item)
            for item in value.get("energies", ())
            if item is not None
        ],
        "energy_cards": _card_ids(value.get("energyCards", ())),
        "tools": _card_ids(value.get("tools", ())),
        "pre_evolution": _card_ids(value.get("preEvolution", ())),
    }


def _public_player_view(player: Any) -> dict[str, Any]:
    if not isinstance(player, Mapping):
        player = {}
    return {
        "active": [_pokemon_view(item) for item in player.get("active", ())],
        "bench": [_pokemon_view(item) for item in player.get("bench", ())],
        "bench_max": _integer(player.get("benchMax")),
        "deck_count": _integer(player.get("deckCount")),
        "discard": _card_ids(player.get("discard", ())),
        # Prize identities and hand identities are intentionally excluded from
        # the common/public state, even when a truth-level test injects them.
        "prize_count": len(player.get("prize", ()))
        if isinstance(player.get("prize", ()), Sequence)
        else 0,
        "hand_count": _integer(player.get("handCount")),
        "poisoned": bool(player.get("poisoned", False)),
        "burned": bool(player.get("burned", False)),
        "asleep": bool(player.get("asleep", False)),
        "paralyzed": bool(player.get("paralyzed", False)),
        "confused": bool(player.get("confused", False)),
    }


def _option_view(option: Any) -> dict[str, Any]:
    if not isinstance(option, Mapping):
        return {}
    return {
        field_name: option[field_name]
        for field_name in OPTION_SEMANTIC_FIELDS
        if field_name in option and option[field_name] is not None
    }


def _select_private_view(select: Any) -> dict[str, Any]:
    if not isinstance(select, Mapping):
        return {}
    return {
        "type": _integer(select.get("type"), -1),
        "context": _integer(select.get("context"), -1),
        "min_count": _integer(select.get("minCount")),
        "max_count": _integer(select.get("maxCount")),
        "remain_damage_counter": _integer(select.get("remainDamageCounter")),
        "remain_energy_cost": _integer(select.get("remainEnergyCost")),
        "options": [_option_view(item) for item in select.get("option", ())],
        "visible_deck": _card_ids(select.get("deck", ())),
        "context_card": _card_id(select.get("contextCard")),
        "effect_card": _card_id(select.get("effect")),
    }


def sanitize_official_public_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Allow-list common state; logs, serials, private cards, and search IDs are absent."""

    current = observation.get("current")
    if not isinstance(current, Mapping):
        current = {}
    players = current.get("players", ())
    if not isinstance(players, Sequence):
        players = ()
    return {
        "turn": _integer(current.get("turn")),
        "turn_action_count": _integer(current.get("turnActionCount")),
        "acting_player": _integer(current.get("yourIndex")),
        "first_player": _integer(current.get("firstPlayer"), -1),
        "supporter_played": bool(current.get("supporterPlayed", False)),
        "stadium_played": bool(current.get("stadiumPlayed", False)),
        "energy_attached": bool(current.get("energyAttached", False)),
        "retreated": bool(current.get("retreated", False)),
        "result": _integer(current.get("result"), -1),
        "stadium": _card_ids(current.get("stadium", ())),
        # Looking cards are agent-visible but may be private to the acting player,
        # so only the count belongs to the common state.
        "looking_count": len(current.get("looking", ()))
        if isinstance(current.get("looking"), Sequence)
        else 0,
        "players": [_public_player_view(player) for player in players[:2]],
    }


def official_private_observation_payload(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        current = {}
    players = current.get("players", ())
    acting = _integer(current.get("yourIndex"))
    own = (
        players[acting]
        if isinstance(players, Sequence)
        and 0 <= acting < len(players)
        and isinstance(players[acting], Mapping)
        else {}
    )
    return {
        "own_hand": _card_ids(own.get("hand", ())),
        "own_looking": _card_ids(current.get("looking", ())),
        "selection": _select_private_view(observation.get("select")),
    }


@dataclass(frozen=True, slots=True)
class OfficialBridgeConfig:
    maximum_complete_actions: int = 4_096

    def __post_init__(self) -> None:
        if self.maximum_complete_actions <= 0:
            raise ValueError("maximum_complete_actions must be positive")


class UnliftableOfficialDecision(RuntimeError):
    pass


def complete_selection_count(contract: SelectionContract) -> int:
    return sum(
        perm(contract.option_count, count)
        for count in range(contract.min_count, contract.max_count + 1)
    )


def complete_official_actions(
    observation: Mapping[str, Any],
    *,
    maximum_complete_actions: int = 4_096,
) -> tuple[AbstractAction, ...]:
    """Enumerate every contract-valid ordered native selection.

    This is shared by the one-step bridge and the production-disabled native
    search game.  Enumeration does not execute an action and does not itself
    imply that a branchable information-set model is faithful.
    """

    if maximum_complete_actions <= 0:
        raise ValueError("maximum_complete_actions must be positive")
    contract = SelectionContract.from_observation(observation)
    total = complete_selection_count(contract)
    if total > maximum_complete_actions:
        raise UnliftableOfficialDecision(
            f"complete ordered selection space {total} exceeds cap "
            f"{maximum_complete_actions}"
        )
    options = observation.get("select", {}).get("option", ())
    actions = []
    for count in range(contract.min_count, contract.max_count + 1):
        for selected in permutations(range(contract.option_count), count):
            option_types = tuple(
                _integer(options[index].get("type"), -1)
                if isinstance(options[index], Mapping)
                else -1
                for index in selected
            )
            action_id = (
                "native:none"
                if not selected
                else "native:" + ",".join(str(index) for index in selected)
            )
            actions.append(
                AbstractAction(
                    action_id=action_id,
                    label=(
                        "select []"
                        if not selected
                        else f"select {list(selected)} types={list(option_types)}"
                    ),
                    official_option_indices=tuple(selected),
                    features=(
                        ("selection_count", float(len(selected))),
                        ("option_type_sum", float(sum(option_types))),
                    ),
                )
            )
    if len(actions) != total:
        raise RuntimeError("complete native action enumeration count mismatch")
    return tuple(actions)


class OfficialDecisionBridge:
    """Complete one-step information-set/action lift over native legal options."""

    def __init__(
        self,
        *,
        ruleset_fingerprint: str,
        official_engine_executed: bool,
        config: OfficialBridgeConfig | None = None,
    ) -> None:
        _validate_digest(ruleset_fingerprint, "ruleset_fingerprint")
        self.config = config or OfficialBridgeConfig()
        self._provenance = GameBoundaryProvenance(
            name="official_agent_visible_complete_selection_bridge",
            kind=GameBoundaryKind.OFFICIAL_ENGINE,
            schema_version=OFFICIAL_BRIDGE_SCHEMA_VERSION,
            ruleset_fingerprint=ruleset_fingerprint,
            official_engine_executed=official_engine_executed,
            fidelity_notes=(
                "One-step native observation/legal-selection lift only. It is "
                "not a transition model, not RocketPrizeDuel, and not evidence "
                "that a full-game neural blueprint is trained."
            ),
        )

    @property
    def provenance(self) -> GameBoundaryProvenance:
        return self._provenance

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema": OFFICIAL_BRIDGE_SCHEMA_VERSION,
                "provenance": asdict(self.provenance),
                "config": asdict(self.config),
                "option_fields": sorted(OPTION_FIELDS),
                "public_state_allowlist": "official_bridge_v1",
            }
        )

    def public_belief_from_official_observation(
        self,
        observation: Mapping[str, Any],
    ) -> PublicBeliefState:
        public = sanitize_official_public_state(observation)
        current = observation.get("current")
        current = current if isinstance(current, Mapping) else {}
        acting = _integer(current.get("yourIndex"))
        players = public["players"]
        own = players[acting] if 0 <= acting < len(players) else {}
        opponent = players[1 - acting] if len(players) == 2 else {}
        public_features = (
            ("turn", float(public["turn"])),
            ("turn_action_count", float(public["turn_action_count"])),
            ("own_board_count", float(len(own.get("active", ())) + len(own.get("bench", ())))),
            ("opponent_board_count", float(len(opponent.get("active", ())) + len(opponent.get("bench", ())))),
            ("own_hand_count", float(own.get("hand_count", 0))),
            ("opponent_hand_count", float(opponent.get("hand_count", 0))),
            ("own_prize_count", float(own.get("prize_count", 0))),
            ("opponent_prize_count", float(opponent.get("prize_count", 0))),
            ("own_deck_count", float(own.get("deck_count", 0))),
            ("opponent_deck_count", float(opponent.get("deck_count", 0))),
        )
        public_state_fingerprint = _fingerprint(public)
        public_history_fingerprint = _fingerprint(
            {
                "public_state": public_state_fingerprint,
                "turn": public["turn"],
                "turn_action_count": public["turn_action_count"],
                "logs_intentionally_omitted": True,
            }
        )
        return PublicBeliefState(
            public_state_fingerprint=public_state_fingerprint,
            public_history_fingerprint=public_history_fingerprint,
            turn=max(0, int(public["turn"])),
            acting_player=InformationSetActor.SELF,
            opponent_range=OpponentRange.normalized(
                (
                    RangeHypothesis(
                        hypothesis_id="opponent:unresolved_public_range",
                        weight=1.0,
                        hidden_resource_counts=(),
                        intent_id="unknown",
                        style_id="unknown",
                        source=(
                            "aggregate placeholder from agent-visible official "
                            "observation; contains no simulator hidden truth"
                        ),
                    ),
                )
            ),
            public_features=public_features,
            updater_fingerprint=self.fingerprint,
        )

    def _complete_actions(
        self,
        observation: Mapping[str, Any],
    ) -> tuple[AbstractAction, ...]:
        return complete_official_actions(
            observation,
            maximum_complete_actions=self.config.maximum_complete_actions,
        )

    def information_set_from_official_observation(
        self,
        observation: Mapping[str, Any],
        belief: PublicBeliefState,
    ) -> InformationSet:
        expected = self.public_belief_from_official_observation(observation)
        if belief.fingerprint != expected.fingerprint:
            raise ValueError("official observation and supplied public belief disagree")
        contract = SelectionContract.from_observation(observation)
        private_fingerprint = _fingerprint(
            official_private_observation_payload(observation)
        )
        actions = self._complete_actions(observation)
        current = observation.get("current")
        current = current if isinstance(current, Mapping) else {}
        identity = _fingerprint(
            {
                "public_belief": belief.fingerprint,
                "private_observation": private_fingerprint,
                "contract": asdict(contract),
                "actions": [
                    {
                        "id": action.action_id,
                        "indices": action.official_option_indices,
                    }
                    for action in actions
                ],
            }
        )
        return InformationSet(
            info_set_id=f"official:{identity}",
            public_belief_fingerprint=belief.fingerprint,
            actor=InformationSetActor.SELF,
            legal_actions=actions,
            depth=max(0, _integer(current.get("turnActionCount"))),
            terminal=False,
            private_observation_fingerprint=private_fingerprint,
        )

    def official_selection(
        self,
        strategy: ActionDistribution,
        observation: Mapping[str, Any],
    ) -> list[int]:
        belief = self.public_belief_from_official_observation(observation)
        information_set = self.information_set_from_official_observation(
            observation,
            belief,
        )
        return self.lift_from_information_set(
            strategy,
            information_set,
            observation,
        )

    @staticmethod
    def lift_from_information_set(
        strategy: ActionDistribution,
        information_set: InformationSet,
        observation: Mapping[str, Any],
    ) -> list[int]:
        """Lift with a previously validated information set, avoiding recomputation."""

        strategy.validate_support(information_set)
        selected = next(
            action.official_option_indices
            for action in information_set.legal_actions
            if action.action_id == strategy.argmax
        )
        return list(SelectionContract.from_observation(observation).validate(selected))

    def one_hot_selection_strategy(
        self,
        information_set: InformationSet,
        selection: Sequence[int],
    ) -> ActionDistribution:
        selected = tuple(int(index) for index in selection)
        matches = [
            action
            for action in information_set.legal_actions
            if action.official_option_indices == selected
        ]
        if len(matches) != 1:
            raise UnliftableOfficialDecision(
                "native selection is not represented by exactly one abstract action"
            )
        selected_id = matches[0].action_id
        return ActionDistribution(
            tuple(
                (
                    action.action_id,
                    1.0 if action.action_id == selected_id else 0.0,
                )
                for action in information_set.legal_actions
            )
        )


@dataclass(frozen=True, slots=True)
class OfficialBridgeDecisionAudit:
    match_id: str
    decision_index: int
    turn: int
    select_type: int
    select_context: int
    option_count: int
    min_count: int
    max_count: int
    complete_action_count: int
    selection: tuple[int, ...]
    lifted_selection: tuple[int, ...]
    information_set_id: str
    public_belief_fingerprint: str
    private_observation_fingerprint: str
    selection_roundtrip_exact: bool
    every_abstract_action_contract_valid: bool
    hidden_canary_invariant: bool
    own_private_sensitivity: bool | None
    unknown_option_fields: tuple[str, ...]
    bridge_latency_ms: float
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.match_id or self.decision_index < 0:
            raise ValueError("bridge decision identity is invalid")
        for field_name in (
            "option_count",
            "min_count",
            "max_count",
            "complete_action_count",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.bridge_latency_ms < 0:
            raise ValueError("bridge latency cannot be negative")
        for field_name in (
            "public_belief_fingerprint",
            "private_observation_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)


def _hidden_canary_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    mutated = deepcopy(dict(observation))
    current = mutated.get("current")
    if isinstance(current, Mapping):
        current = dict(current)
        mutated["current"] = current
        players = list(current.get("players", ()))
        acting = _integer(current.get("yourIndex"))
        opponent = 1 - acting
        if 0 <= opponent < len(players) and isinstance(players[opponent], Mapping):
            player = dict(players[opponent])
            player["hand"] = [
                {
                    "id": HIDDEN_CANARY_CARD_ID,
                    "serial": HIDDEN_CANARY_CARD_ID,
                    "playerIndex": opponent,
                }
            ]
            prize = list(player.get("prize", ()))
            if prize:
                prize[0] = {
                    "id": HIDDEN_CANARY_CARD_ID,
                    "serial": HIDDEN_CANARY_CARD_ID,
                    "playerIndex": opponent,
                }
            player["prize"] = prize
            players[opponent] = player
            current["players"] = players
    mutated["logs"] = [
        {
            "type": 999,
            "cardId": HIDDEN_CANARY_CARD_ID,
            "serial": HIDDEN_CANARY_CARD_ID,
        }
    ]
    mutated["search_begin_input"] = f"hidden-canary-{HIDDEN_CANARY_CARD_ID}"
    return mutated


def audit_official_bridge_decision(
    bridge: OfficialDecisionBridge,
    observation: Mapping[str, Any],
    selection: Sequence[int],
    *,
    match_id: str,
    decision_index: int,
) -> OfficialBridgeDecisionAudit:
    started = perf_counter()
    contract = SelectionContract.from_observation(observation)
    select = observation.get("select")
    select = select if isinstance(select, Mapping) else {}
    unknown_option_fields = sorted(
        {
            str(field_name)
            for option in select.get("option", ())
            if isinstance(option, Mapping)
            for field_name in set(option) - OPTION_FIELDS
        }
    )
    belief = bridge.public_belief_from_official_observation(observation)
    information_set = bridge.information_set_from_official_observation(
        observation,
        belief,
    )
    strategy = bridge.one_hot_selection_strategy(information_set, selection)
    lifted = tuple(
        bridge.lift_from_information_set(strategy, information_set, observation)
    )
    every_valid = True
    for action in information_set.legal_actions:
        try:
            contract.validate(action.official_option_indices)
        except ValueError:
            every_valid = False
            break
    canary = _hidden_canary_observation(observation)
    canary_belief = bridge.public_belief_from_official_observation(canary)
    hidden_invariant = (
        canary_belief.fingerprint == belief.fingerprint
        and _fingerprint(official_private_observation_payload(canary))
        == information_set.private_observation_fingerprint
    )
    private_sensitivity: bool | None = None
    current = observation.get("current")
    if isinstance(current, Mapping):
        players = current.get("players", ())
        acting = _integer(current.get("yourIndex"))
        if (
            isinstance(players, Sequence)
            and 0 <= acting < len(players)
            and isinstance(players[acting], Mapping)
            and players[acting].get("hand")
        ):
            own_mutated = deepcopy(dict(observation))
            own_current = dict(own_mutated["current"])
            own_mutated["current"] = own_current
            own_players = list(own_current["players"])
            own_player = dict(own_players[acting])
            own_hand = list(own_player["hand"])
            own_card = dict(own_hand[0])
            own_card["id"] = HIDDEN_CANARY_CARD_ID
            own_hand[0] = own_card
            own_player["hand"] = own_hand
            own_players[acting] = own_player
            own_current["players"] = own_players
            own_belief = bridge.public_belief_from_official_observation(own_mutated)
            private_sensitivity = (
                own_belief.fingerprint == belief.fingerprint
                and _fingerprint(official_private_observation_payload(own_mutated))
                != information_set.private_observation_fingerprint
            )
    return OfficialBridgeDecisionAudit(
        match_id=match_id,
        decision_index=decision_index,
        turn=_integer(
            observation.get("current", {}).get("turn")
            if isinstance(observation.get("current"), Mapping)
            else 0
        ),
        select_type=_integer(select.get("type"), -1),
        select_context=_integer(select.get("context"), -1),
        option_count=contract.option_count,
        min_count=contract.min_count,
        max_count=contract.max_count,
        complete_action_count=len(information_set.legal_actions),
        selection=tuple(int(index) for index in selection),
        lifted_selection=lifted,
        information_set_id=information_set.info_set_id,
        public_belief_fingerprint=belief.fingerprint,
        private_observation_fingerprint=(
            information_set.private_observation_fingerprint or ""
        ),
        selection_roundtrip_exact=tuple(selection) == lifted,
        every_abstract_action_contract_valid=every_valid,
        hidden_canary_invariant=hidden_invariant,
        own_private_sensitivity=private_sensitivity,
        unknown_option_fields=tuple(unknown_option_fields),
        bridge_latency_ms=(perf_counter() - started) * 1_000.0,
    )


class BridgeAuditedCompetitionAgent:
    """Transparent shadow wrapper: delegates first and never replaces selection."""

    def __init__(
        self,
        delegate: Any,
        bridge: OfficialDecisionBridge,
        *,
        match_id: str,
    ) -> None:
        self.delegate = delegate
        self.bridge = bridge
        self.match_id = match_id
        self.audit_records: list[OfficialBridgeDecisionAudit] = []

    def reset(self) -> None:
        self.audit_records.clear()
        self.delegate.reset()

    def choose(self, observation: Mapping[str, Any]) -> list[int]:
        selection = self.delegate.choose(observation)
        try:
            audit = audit_official_bridge_decision(
                self.bridge,
                observation,
                selection,
                match_id=self.match_id,
                decision_index=len(self.audit_records),
            )
        except Exception as error:  # shadow audit must never break official play
            contract = SelectionContract.from_observation(observation)
            select = observation.get("select")
            select = select if isinstance(select, Mapping) else {}
            audit = OfficialBridgeDecisionAudit(
                match_id=self.match_id,
                decision_index=len(self.audit_records),
                turn=0,
                select_type=_integer(select.get("type"), -1),
                select_context=_integer(select.get("context"), -1),
                option_count=contract.option_count,
                min_count=contract.min_count,
                max_count=contract.max_count,
                complete_action_count=complete_selection_count(contract),
                selection=tuple(selection),
                lifted_selection=(),
                information_set_id="official:error",
                public_belief_fingerprint="0" * 64,
                private_observation_fingerprint="0" * 64,
                selection_roundtrip_exact=False,
                every_abstract_action_contract_valid=False,
                hidden_canary_invariant=False,
                own_private_sensitivity=None,
                unknown_option_fields=(),
                bridge_latency_ms=0.0,
                error=f"{type(error).__name__}: {error}",
            )
        self.audit_records.append(audit)
        # The audited bridge is shadow-only; the delegate's exact list is the
        # only list returned to the native engine.
        return list(selection)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)
