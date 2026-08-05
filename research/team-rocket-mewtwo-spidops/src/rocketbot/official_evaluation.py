"""Local evaluation harness for the competition-only native PTCG engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from statistics import fmean
import sys
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence

from .competition import (
    ARCHETYPE_SIGNATURES,
    DeckResolution,
    OptionType,
    Phase6CompetitionAgent,
    SelectionContract,
    _in_play,
    _player,
    _safe_int,
    _state,
)
from .competition_search import ABOMASNOW_PRIOR
from .evaluation import OfficialMatchRecord


OFFICIAL_EVALUATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class KaggleDeckProfile:
    label: str
    archetype: str
    card_ids: tuple[int, ...]
    provenance: str

    def __post_init__(self) -> None:
        if not self.label or not self.archetype or not self.provenance:
            raise ValueError("deck profile metadata must be non-empty")
        if len(self.card_ids) != 60:
            raise ValueError("official deck profile must contain exactly 60 IDs")
        if any(not isinstance(value, int) or value <= 0 for value in self.card_ids):
            raise ValueError("official deck IDs must be positive integers")

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema": OFFICIAL_EVALUATION_SCHEMA_VERSION,
            "label": self.label,
            "archetype": self.archetype,
            "card_ids": self.card_ids,
            "provenance": self.provenance,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def rocket_compatibility_profile(deck: DeckResolution) -> KaggleDeckProfile:
    if not deck.playable:
        raise ValueError("Rocket compatibility profile requires a playable resolution")
    return KaggleDeckProfile(
        label="rocket_mewtwo_watchtower_compatibility",
        archetype="rocket_mewtwo",
        card_ids=deck.card_ids,
        provenance=(
            "Exact NAIC list except explicit Prism Tower -> Team Rocket's Watchtower "
            "compatibility substitution for the TEF-POR Kaggle card pool."
        ),
    )


TRAINER_SHELL: tuple[int, ...] = tuple(
    [1227] * 4 + [1121] * 4 + [1079] * 4 + [1123] * 4 + [1152] * 4
)


OFFICIAL_PANEL_DECKS: tuple[KaggleDeckProfile, ...] = (
    KaggleDeckProfile(
        label="official_sample_abomasnow",
        archetype="abomasnow",
        card_ids=ABOMASNOW_PRIOR,
        provenance="Official Kaggle sample_submission/deck.csv.",
    ),
    KaggleDeckProfile(
        label="alakazam_legality_probe",
        archetype="alakazam",
        card_ids=tuple([741] * 4 + [742] * 4 + [743] * 4) + TRAINER_SHELL + tuple([5] * 28),
        provenance="Deterministic legal archetype probe; not a tournament-optimized list.",
    ),
    KaggleDeckProfile(
        label="starmie_legality_probe",
        archetype="starmie",
        card_ids=tuple([1030] * 4 + [1031] * 4) + TRAINER_SHELL + tuple([3] * 32),
        provenance="Deterministic legal archetype probe; not a tournament-optimized list.",
    ),
    KaggleDeckProfile(
        label="dragapult_legality_probe",
        archetype="dragapult",
        card_ids=(
            tuple([119] * 4 + [120] * 4 + [121] * 4)
            + TRAINER_SHELL
            + tuple([2] * 14 + [5] * 14)
        ),
        provenance="Deterministic legal archetype probe; not a tournament-optimized list.",
    ),
    KaggleDeckProfile(
        label="crustle_legality_probe",
        archetype="crustle",
        card_ids=tuple([532] * 4 + [533] * 4) + TRAINER_SHELL + tuple([6] * 32),
        provenance="Deterministic legal archetype probe; not a tournament-optimized list.",
    ),
)


class OfficialPolicy(Protocol):
    def reset(self) -> None: ...

    def choose(self, observation: Mapping[str, Any]) -> list[int]: ...


class PriorityBaseline:
    """Dependency-free legal opponent with a configurable main-action style."""

    PRIORITIES: Mapping[str, Mapping[int, float]] = {
        "tempo": {
            OptionType.EVOLVE: 190,
            OptionType.ATTACH: 180,
            OptionType.ABILITY: 170,
            OptionType.PLAY: 160,
            OptionType.ATTACK: 100,
            OptionType.RETREAT: 20,
            OptionType.END: 0,
        },
        "attack_rush": {
            OptionType.ATTACK: 220,
            OptionType.ATTACH: 185,
            OptionType.EVOLVE: 175,
            OptionType.ABILITY: 130,
            OptionType.PLAY: 115,
            OptionType.RETREAT: 30,
            OptionType.END: 0,
        },
        "control": {
            OptionType.PLAY: 205,
            OptionType.ABILITY: 190,
            OptionType.EVOLVE: 175,
            OptionType.ATTACH: 165,
            OptionType.RETREAT: 125,
            OptionType.ATTACK: 105,
            OptionType.END: 0,
        },
    }

    def __init__(self, deck: Sequence[int], *, style: str = "tempo", reverse: bool = False) -> None:
        if len(deck) != 60:
            raise ValueError("baseline deck must contain 60 IDs")
        if style not in self.PRIORITIES:
            raise ValueError(f"unknown baseline style: {style}")
        self.deck = tuple(int(value) for value in deck)
        self.style = style
        self.reverse = reverse

    def reset(self) -> None:
        return None

    def choose(self, observation: Mapping[str, Any]) -> list[int]:
        select = observation.get("select")
        if select is None:
            return list(self.deck)
        contract = SelectionContract.from_observation(observation)
        options = select.get("option", ())
        context = _safe_int(select.get("context"), -1)
        if context == 0:
            priorities = self.PRIORITIES[self.style]
            ranked = sorted(
                range(len(options)),
                key=lambda index: (
                    -priorities.get(_safe_int(options[index].get("type"), -1), 50.0),
                    -index if self.reverse else index,
                ),
            )
        elif _safe_int(select.get("type"), -1) == 9:  # YES_NO
            ranked = sorted(
                range(len(options)),
                key=lambda index: (
                    0 if _safe_int(options[index].get("type"), -1) == OptionType.YES else 1,
                    index,
                ),
            )
        elif _safe_int(select.get("type"), -1) == 8:  # COUNT
            ranked = sorted(
                range(len(options)),
                key=lambda index: -_safe_int(options[index].get("number"), 0),
            )
        else:
            ranked = list(range(len(options)))
            if self.reverse:
                ranked.reverse()
        return ranked[: contract.max_count]


class SeededRandomBaseline:
    def __init__(self, deck: Sequence[int], seed: int) -> None:
        if len(deck) != 60:
            raise ValueError("baseline deck must contain 60 IDs")
        self.deck = tuple(int(value) for value in deck)
        self.seed = int(seed)
        self.random = random.Random(self.seed)

    def reset(self) -> None:
        self.random.seed(self.seed)

    def choose(self, observation: Mapping[str, Any]) -> list[int]:
        select = observation.get("select")
        if select is None:
            return list(self.deck)
        contract = SelectionContract.from_observation(observation)
        return self.random.sample(range(contract.option_count), contract.max_count)


@dataclass(frozen=True, slots=True)
class OfficialMatchExecution:
    record: OfficialMatchRecord
    replay_data: str
    agent_summary: Mapping[str, Any]


class OfficialMatchRunner:
    """Sequential runner around the native engine's process-global battle state."""

    def __init__(
        self,
        runtime_directory: str | Path,
        *,
        max_selections: int = 2000,
    ) -> None:
        self.runtime_directory = Path(runtime_directory).resolve()
        if not (self.runtime_directory / "cg").is_dir():
            raise FileNotFoundError(
                f"official runtime must contain cg/: {self.runtime_directory}"
            )
        if max_selections <= 0:
            raise ValueError("max_selections must be positive")
        self.max_selections = max_selections
        path_text = str(self.runtime_directory)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
        from cg.game import (  # type: ignore[import-not-found]
            battle_finish,
            battle_select,
            battle_start,
            visualize_data,
        )

        self._battle_finish = battle_finish
        self._battle_select = battle_select
        self._battle_start = battle_start
        self._visualize_data = visualize_data

    def validate_deck(self, profile: KaggleDeckProfile) -> tuple[bool, int, int]:
        observation, start = self._battle_start(list(profile.card_ids), list(profile.card_ids))
        try:
            return (observation is not None, int(start.errorPlayer), int(start.errorType))
        finally:
            if observation is not None:
                self._battle_finish()

    def run_match(
        self,
        *,
        match_id: str,
        target_agent: Phase6CompetitionAgent,
        target_deck: KaggleDeckProfile,
        opponent_agent: OfficialPolicy,
        opponent_deck: KaggleDeckProfile,
        opponent_style: str,
        target_seat: int,
        capture_replay: bool = True,
    ) -> OfficialMatchExecution:
        if target_seat not in (0, 1):
            raise ValueError("target_seat must be 0 or 1")
        target_agent.reset()
        opponent_agent.reset()
        agents: list[OfficialPolicy] = [target_agent, opponent_agent]
        decks = [target_deck, opponent_deck]
        if target_seat == 1:
            agents.reverse()
            decks.reverse()
        observation, start = self._battle_start(
            list(decks[0].card_ids), list(decks[1].card_ids)
        )
        if observation is None:
            raise RuntimeError(
                f"official engine rejected deck: player={start.errorPlayer}, "
                f"error={start.errorType}"
            )

        selections = 0
        runner_fallback_count = 0
        illegal_action_count = 0
        soft_timeout_count = 0
        action_counts: Counter[str] = Counter()
        latencies: list[float] = []
        lethal_opportunities = 0
        lethal_conversions = 0
        first_attack_turn: int | None = None
        first_prize_turn: int | None = None
        board_samples: list[tuple[int, int, int, int, int, int]] = []
        initial_own_prizes = len(_player(observation, target_seat).get("prize", ()))
        engine_error: str | None = None
        replay_data = ""

        try:
            while (
                observation is not None
                and _safe_int(_state(observation).get("result"), -1) == -1
                and selections < self.max_selections
            ):
                actor = _safe_int(_state(observation).get("yourIndex"), 0)
                is_target = actor == target_seat
                if is_target:
                    own = _player(observation, target_seat)
                    opposing = _player(observation, 1 - target_seat)
                    own_board = _in_play(own)
                    opposing_board = _in_play(opposing)
                    board_samples.append(
                        (
                            len(own_board),
                            len(opposing_board),
                            sum(_safe_int(card.get("hp"), 0) for card in own_board),
                            sum(_safe_int(card.get("hp"), 0) for card in opposing_board),
                            _safe_int(own.get("handCount"), len(own.get("hand", ()))),
                            len(observation.get("select", {}).get("option", ())),
                        )
                    )
                    if (
                        first_prize_turn is None
                        and len(own.get("prize", ())) < initial_own_prizes
                    ):
                        first_prize_turn = _safe_int(_state(observation).get("turn"), 0)
                    lethal_indices = self._lethal_attack_indices(target_agent, observation)
                    lethal_opportunities += int(bool(lethal_indices))

                contract = SelectionContract.from_observation(observation)
                started = perf_counter()
                try:
                    selection = agents[actor].choose(observation)
                    normalized = contract.validate(selection)
                except Exception:
                    if is_target:
                        illegal_action_count += 1
                        runner_fallback_count += 1
                    normalized = contract.fallback()
                latency_ms = (perf_counter() - started) * 1000.0
                if is_target:
                    latencies.append(latency_ms)
                    if latency_ms > target_agent.config.decision_budget_ms:
                        soft_timeout_count += 1
                    selected_options = [
                        observation["select"]["option"][index] for index in normalized
                    ]
                    for option in selected_options:
                        raw_type = _safe_int(option.get("type"), -1)
                        try:
                            action_counts[OptionType(raw_type).name.lower()] += 1
                        except ValueError:
                            action_counts[f"unknown_{raw_type}"] += 1
                    attack_indices = {
                        index
                        for index in normalized
                        if _safe_int(
                            observation["select"]["option"][index].get("type"), -1
                        )
                        == OptionType.ATTACK
                    }
                    if attack_indices and first_attack_turn is None:
                        first_attack_turn = _safe_int(_state(observation).get("turn"), 0)
                    if lethal_indices and attack_indices.intersection(lethal_indices):
                        lethal_conversions += 1

                observation = self._battle_select(list(normalized))
                selections += 1

            if selections >= self.max_selections:
                engine_error = f"selection cap reached ({self.max_selections})"
            if capture_replay:
                replay_data = self._visualize_data()
            final_state = _state(observation or {})
            winner = _safe_int(final_state.get("result"), -1)
            turns = _safe_int(final_state.get("turn"), 0)
            own_final = _player(observation or {}, target_seat)
            opposing_final = _player(observation or {}, 1 - target_seat)
        except Exception as exc:
            engine_error = f"{type(exc).__name__}: {exc}"
            final_state = _state(observation or {})
            winner = _safe_int(final_state.get("result"), -1)
            turns = _safe_int(final_state.get("turn"), 0)
            own_final = _player(observation or {}, target_seat)
            opposing_final = _player(observation or {}, 1 - target_seat)
        finally:
            self._battle_finish()

        telemetry = tuple(target_agent.memory.telemetry)
        strategic_values = tuple(
            fmean(item.strategic.values[index] for item in telemetry)
            if telemetry
            else 0.0
            for index in range(6)
        )
        strategic_weights_raw = tuple(
            fmean(item.strategic.weights[index] for item in telemetry)
            if telemetry
            else 1.0 / 6.0
            for index in range(6)
        )
        weight_total = sum(strategic_weights_raw)
        strategic_weights = tuple(value / weight_total for value in strategic_weights_raw)
        strategic_aggregate = (
            fmean(item.strategic.aggregate for item in telemetry) if telemetry else 0.0
        )
        belief_nll, belief_brier = self._belief_metrics(
            telemetry, opponent_deck.archetype
        )
        searched = tuple(item for item in telemetry if item.search_used)
        search_bonuses = [
            max((value for _, value in item.search_bonus), default=0.0)
            for item in searched
        ]

        def sample_mean(index: int) -> float:
            return fmean(sample[index] for sample in board_samples) if board_samples else 0.0

        record = OfficialMatchRecord(
            match_id=match_id,
            agent_label=target_deck.label,
            ablation=target_agent.config.ablation.value,
            opponent_label=opponent_deck.label,
            opponent_archetype=opponent_deck.archetype,
            opponent_style=opponent_style,
            agent_seat=target_seat,
            winner=winner,
            turns=turns,
            selections=selections,
            decision_count=len(telemetry),
            runner_fallback_count=runner_fallback_count,
            illegal_action_count=illegal_action_count,
            soft_timeout_count=soft_timeout_count,
            mean_latency_ms=fmean(latencies) if latencies else 0.0,
            max_latency_ms=max(latencies, default=0.0),
            action_counts=tuple(sorted(action_counts.items())),
            lethal_opportunities=lethal_opportunities,
            lethal_conversions=lethal_conversions,
            first_attack_turn=first_attack_turn,
            first_prize_turn=first_prize_turn,
            own_prizes_remaining=len(own_final.get("prize", ())),
            opponent_prizes_remaining=len(opposing_final.get("prize", ())),
            mean_own_board=sample_mean(0),
            mean_opponent_board=sample_mean(1),
            mean_own_hp=sample_mean(2),
            mean_opponent_hp=sample_mean(3),
            mean_own_hand=sample_mean(4),
            mean_option_count=sample_mean(5),
            strategic_values=strategic_values,
            strategic_weights=strategic_weights,
            strategic_aggregate=strategic_aggregate,
            belief_nll=belief_nll,
            belief_brier=belief_brier,
            search_invocations=len(searched),
            search_changes=sum(item.search_changed_action for item in searched),
            mean_search_bonus=fmean(search_bonuses) if search_bonuses else 0.0,
            engine_error=engine_error,
        )
        return OfficialMatchExecution(
            record=record,
            replay_data=replay_data,
            agent_summary=target_agent.summary(),
        )

    @staticmethod
    def _lethal_attack_indices(
        target_agent: Phase6CompetitionAgent,
        observation: Mapping[str, Any],
    ) -> set[int]:
        options = observation.get("select", {}).get("option", ())
        opponent = _player(observation, 1 - _safe_int(_state(observation).get("yourIndex"), 0))
        active = _in_play(opponent)[:1]
        if not active:
            return set()
        hp = _safe_int(active[0].get("hp"), 9999)
        result: set[int] = set()
        for index, option in enumerate(options):
            if _safe_int(option.get("type"), -1) != OptionType.ATTACK:
                continue
            attack_id = _safe_int(option.get("attackId"), -1)
            if target_agent._expected_attack_damage(observation, attack_id) >= hp:
                result.add(index)
        return result

    @staticmethod
    def _belief_metrics(telemetry: Sequence[Any], true_archetype: str) -> tuple[float, float]:
        if not telemetry or true_archetype not in ARCHETYPE_SIGNATURES:
            return (math.log(len(ARCHETYPE_SIGNATURES)), 1.0)
        nlls: list[float] = []
        briers: list[float] = []
        for item in telemetry:
            posterior = dict(item.archetype_posterior)
            nlls.append(-math.log(max(posterior.get(true_archetype, 0.0), 1e-12)))
            briers.append(
                sum(
                    (posterior.get(name, 0.0) - float(name == true_archetype)) ** 2
                    for name in ARCHETYPE_SIGNATURES
                )
            )
        return (fmean(nlls), fmean(briers))
