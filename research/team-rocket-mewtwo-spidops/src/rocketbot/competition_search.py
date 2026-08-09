"""Bounded integration-baseline look-ahead through the official Kaggle search API.

This adapter is optional and imported lazily: normal research-package imports
do not require the competition-only native library.  Search is fail-soft.  A
bad hidden-state completion, engine error, or elapsed deadline yields no bonus,
leaving the legal deterministic policy untouched.

This sampled rollout is a diagnostic/fallback baseline, not a game-theoretic
blueprint or safe information-set resolver.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from time import perf_counter
from typing import Any, Mapping, Sequence

try:  # Package import during research/evaluation.
    from .competition import (
        AblationMode,
        CompetitionAgentConfig,
        DeckResolution,
        OptionType,
        Phase6CompetitionAgent,
        SearchAdvisor,
        SelectionContract,
        _card_id,
        _in_play,
        _player,
        _safe_int,
        _state,
        _your_index,
        visible_card_ids,
    )
except ImportError:  # Flat import inside the generated compatibility baseline.
    from rocketbot_competition import (  # type: ignore[no-redef]
        AblationMode,
        CompetitionAgentConfig,
        DeckResolution,
        OptionType,
        Phase6CompetitionAgent,
        SearchAdvisor,
        SelectionContract,
        _card_id,
        _in_play,
        _player,
        _safe_int,
        _state,
        _your_index,
        visible_card_ids,
    )


ABOMASNOW_PRIOR: tuple[int, ...] = tuple(
    [1158]
    + [721] * 2
    + [722] * 4
    + [723] * 4
    + [1145] * 4
    + [1205] * 2
    + [1227] * 4
    + [1235] * 4
    + [3] * 35
)


class OfficialCgSearchAdvisor(SearchAdvisor):
    """Small, deadline-aware rollouts rooted in official engine state clones."""

    def __init__(
        self,
        own_deck: DeckResolution,
        *,
        opponent_prior: Sequence[int] = ABOMASNOW_PRIOR,
        rollout_steps: int = 7,
        value_scale: float = 0.22,
    ) -> None:
        if not own_deck.playable:
            raise ValueError("search advisor requires a playable own deck")
        if len(opponent_prior) != 60:
            raise ValueError("opponent prior must contain exactly 60 card IDs")
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        if value_scale <= 0:
            raise ValueError("value_scale must be positive")
        self.own_deck = own_deck
        self.opponent_prior = tuple(int(value) for value in opponent_prior)
        self.rollout_steps = rollout_steps
        self.value_scale = value_scale
        self.last_error: str | None = None
        self.last_nodes = 0
        self._rollout_agent = Phase6CompetitionAgent(
            own_deck,
            config=CompetitionAgentConfig(
                ablation=AblationMode.BELIEF_STRATEGIC,
                decision_budget_ms=25.0,
                search_budget_ms=0.0,
            ),
        )

    def advise(
        self,
        observation: Mapping[str, Any],
        candidate_indices: Sequence[int],
        *,
        deadline: float,
        root_player: int,
    ) -> Mapping[int, float]:
        self.last_error = None
        self.last_nodes = 0
        contract = SelectionContract.from_observation(observation)
        # Each root candidate represents one complete selection.  Multi-card
        # combinatorics stay with the autoregressive policy until a later search
        # adapter explicitly enumerates combinations.
        if contract.min_count != 1 or contract.max_count != 1:
            return {}
        candidates = tuple(
            index
            for index in dict.fromkeys(int(value) for value in candidate_indices)
            if 0 <= index < contract.option_count
        )
        if not candidates or perf_counter() >= deadline:
            return {}

        try:
            from cg.api import (  # Competition-only package, loaded on demand.
                search_begin,
                search_end,
                search_release,
                search_step,
                to_observation_class,
            )
        except Exception as exc:
            self.last_error = f"cg import failed: {type(exc).__name__}: {exc}"
            return {}

        root = None
        created_ids: set[int] = set()
        try:
            predictions = self._hidden_predictions(observation)
            root = search_begin(to_observation_class(dict(observation)), **predictions)
            root_value = self._state_value(observation, root_player)
            bonuses: dict[int, float] = {}
            for index in candidates:
                if perf_counter() >= deadline:
                    break
                branch = search_step(root.searchId, [index])
                created_ids.add(branch.searchId)
                self.last_nodes += 1
                branch, branch_ids = self._rollout_branch(
                    branch,
                    root_player=root_player,
                    deadline=deadline,
                    search_step=search_step,
                )
                created_ids.update(branch_ids)
                branch_raw = asdict(branch.observation)
                bonuses[index] = self.value_scale * (
                    self._state_value(branch_raw, root_player) - root_value
                )
            return bonuses
        except Exception as exc:
            self.last_error = f"search failed: {type(exc).__name__}: {exc}"
            return {}
        finally:
            if root is not None:
                try:
                    for search_id in created_ids:
                        if search_id != root.searchId:
                            search_release(search_id)
                except Exception:
                    pass
                try:
                    search_end()
                except Exception:
                    pass

    def _rollout_branch(
        self,
        branch: Any,
        *,
        root_player: int,
        deadline: float,
        search_step: Any,
    ) -> tuple[Any, set[int]]:
        created_ids: set[int] = set()
        start_turn = _safe_int(asdict(branch.observation).get("current", {}).get("turn"), 0)
        for _ in range(self.rollout_steps - 1):
            if perf_counter() >= deadline:
                break
            raw = asdict(branch.observation)
            current = raw.get("current") or {}
            if _safe_int(current.get("result"), -1) != -1 or raw.get("select") is None:
                break
            # Do not model more than one opposing turn.  The short horizon is a
            # tactical validator, not the Phase 3 strategic-plan search.
            if _safe_int(current.get("turn"), start_turn) > start_turn + 1:
                break
            selection = self._rollout_agent.choose(raw)
            branch = search_step(branch.searchId, selection)
            created_ids.add(branch.searchId)
            self.last_nodes += 1
        return branch, created_ids

    def _hidden_predictions(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        your_index = _your_index(observation)
        your = _player(observation, your_index)
        opponent = _player(observation, 1 - your_index)

        your_pool = Counter(self.own_deck.card_ids)
        your_visible_without_prize = self._visible_without_prize(your)
        your_pool.subtract(your_visible_without_prize)
        self._drop_nonpositive(your_pool)
        your_prize = self._complete_prizes(your, your_pool)
        your_deck = self._take_cards(your_pool, _safe_int(your.get("deckCount"), 0))

        opponent_prior = Counter(self.opponent_prior)
        opponent_visible_without_prize = self._visible_without_prize(opponent)
        self._make_prior_cover(opponent_prior, opponent_visible_without_prize)
        opponent_prior.subtract(opponent_visible_without_prize)
        self._drop_nonpositive(opponent_prior)
        opponent_active: list[int] = []
        active = opponent.get("active", ())
        if isinstance(active, Sequence) and active and active[0] is None:
            basic = next(
                (
                    value
                    for value in self.opponent_prior
                    if value in (721, 722, 400, 414, 431, 434)
                    and opponent_prior[value] > 0
                ),
                721,
            )
            opponent_active = [basic]
            opponent_prior[basic] -= 1
            self._drop_nonpositive(opponent_prior)
        opponent_prize = self._complete_prizes(opponent, opponent_prior)
        opponent_hand = self._take_cards(
            opponent_prior, _safe_int(opponent.get("handCount"), 0)
        )
        opponent_deck = self._take_cards(
            opponent_prior, _safe_int(opponent.get("deckCount"), 0)
        )

        return {
            "your_deck": your_deck,
            "your_prize": your_prize,
            "opponent_deck": opponent_deck,
            "opponent_prize": opponent_prize,
            "opponent_hand": opponent_hand,
            "opponent_active": opponent_active,
            "manual_coin": False,
        }

    @staticmethod
    def _visible_without_prize(player: Mapping[str, Any]) -> Counter[int]:
        proxy = dict(player)
        proxy["prize"] = ()
        return visible_card_ids(proxy)

    def _complete_prizes(
        self, player: Mapping[str, Any], pool: Counter[int]
    ) -> list[int]:
        prizes = player.get("prize", ())
        if not isinstance(prizes, Sequence):
            return []
        result: list[int] = []
        unknown_count = 0
        for prize in prizes:
            card_id = _card_id(prize if isinstance(prize, Mapping) else None)
            if card_id is None or card_id < 0:
                unknown_count += 1
            else:
                result.append(card_id)
                pool[card_id] -= 1
        self._drop_nonpositive(pool)
        result.extend(self._take_cards(pool, unknown_count))
        return result

    @staticmethod
    def _drop_nonpositive(pool: Counter[int]) -> None:
        for card_id in tuple(pool):
            if pool[card_id] <= 0:
                del pool[card_id]

    @staticmethod
    def _make_prior_cover(pool: Counter[int], required: Counter[int]) -> None:
        for card_id, count in required.items():
            deficit = count - pool[card_id]
            while deficit > 0:
                donor = max(pool, key=lambda value: pool[value], default=3)
                if pool[donor] <= 0:
                    break
                pool[donor] -= 1
                pool[card_id] += 1
                deficit -= 1
        OfficialCgSearchAdvisor._drop_nonpositive(pool)

    @staticmethod
    def _take_cards(pool: Counter[int], count: int) -> list[int]:
        result: list[int] = []
        # Stable ordering makes hidden completion and search evidence replayable.
        for card_id in sorted(pool, key=lambda value: (-pool[value], value)):
            take = min(pool[card_id], count - len(result))
            result.extend([card_id] * take)
            pool[card_id] -= take
            if len(result) == count:
                break
        if len(result) < count:
            result.extend([3] * (count - len(result)))
        OfficialCgSearchAdvisor._drop_nonpositive(pool)
        return result

    @staticmethod
    def _state_value(observation: Mapping[str, Any], root_player: int) -> float:
        state = _state(observation)
        result = _safe_int(state.get("result"), -1)
        if result != -1:
            if result == 2:
                return 0.0
            return 1200.0 if result == root_player else -1200.0
        own = _player(observation, root_player)
        opponent = _player(observation, 1 - root_player)
        own_board = _in_play(own)
        opposing_board = _in_play(opponent)
        own_hp = sum(_safe_int(card.get("hp"), 0) for card in own_board)
        opposing_hp = sum(_safe_int(card.get("hp"), 0) for card in opposing_board)
        own_energy = sum(len(card.get("energies", ())) for card in own_board)
        opposing_energy = sum(len(card.get("energies", ())) for card in opposing_board)
        own_prizes = len(own.get("prize", ()))
        opposing_prizes = len(opponent.get("prize", ()))
        own_hand = _safe_int(own.get("handCount"), len(own.get("hand", ())))
        opposing_hand = _safe_int(opponent.get("handCount"), 0)
        return (
            135.0 * (opposing_prizes - own_prizes)
            + 0.08 * (own_hp - opposing_hp)
            + 17.0 * (own_energy - opposing_energy)
            + 13.0 * (len(own_board) - len(opposing_board))
            + 2.5 * (own_hand - opposing_hand)
        )
