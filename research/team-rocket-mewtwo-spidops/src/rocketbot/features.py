"""Deterministic strategic features for the Phase 1 baseline encoder."""

from __future__ import annotations

from typing import Iterable

import torch

from .state import GameState, PlayerId, PlayerState, PokemonState


STATE_FEATURE_NAMES: tuple[str, ...] = (
    "turn_progress",
    "is_our_turn",
    "own_prizes_remaining",
    "opponent_prizes_remaining",
    "prize_advantage",
    "own_hand_size",
    "opponent_hand_size",
    "own_deck_fraction",
    "opponent_deck_fraction",
    "own_discard_fraction",
    "opponent_discard_fraction",
    "own_bench_occupancy",
    "opponent_bench_occupancy",
    "own_active_present",
    "opponent_active_present",
    "own_active_hp_fraction",
    "opponent_active_hp_fraction",
    "own_active_damage_pressure",
    "opponent_active_damage_pressure",
    "own_energy_in_play",
    "opponent_energy_in_play",
    "own_tools_in_play",
    "opponent_tools_in_play",
    "energy_attachment_available",
    "supporter_available",
    "retreat_available",
    "stadium_present",
    "own_mewtwo_in_play",
    "own_tarountula_in_play",
    "own_spidops_in_play",
    "own_articuno_in_play",
    "own_mimikyu_in_play",
    "own_clefairy_in_play",
    "own_rocket_energy_in_play",
    "own_grass_energy_in_play",
    "own_psychic_energy_in_play",
    "own_board_slots_open",
)


def _active_hp_fraction(active: PokemonState | None) -> float:
    if active is None:
        return 0.0
    return active.hp_remaining / active.max_hp


def _active_damage_pressure(active: PokemonState | None) -> float:
    if active is None:
        return 0.0
    return min(active.damage / active.max_hp, 2.0)


def _energy_in_play(player: PlayerState) -> int:
    return sum(len(pokemon.attached_energy) for pokemon in player.pokemon_in_play())


def _tools_in_play(player: PlayerState) -> int:
    return sum(pokemon.tool is not None for pokemon in player.pokemon_in_play())


def _pokemon_card_count(player: PlayerState, card_name: str) -> int:
    count = 0
    for pokemon in player.pokemon_in_play():
        cards = (pokemon.top_card,) + pokemon.evolution_stack
        count += sum(card.card_name == card_name for card in cards)
    return count


def _attached_energy_count(player: PlayerState, card_name: str) -> int:
    return sum(
        energy.card_name == card_name
        for pokemon in player.pokemon_in_play()
        for energy in pokemon.attached_energy
    )


class StateTensorizer:
    """Maps complete states to a stable, version-1 strategic feature tensor."""

    feature_names = STATE_FEATURE_NAMES
    feature_dim = len(STATE_FEATURE_NAMES)

    def tensorize_one(
        self,
        state: GameState,
        *,
        perspective: PlayerId = PlayerId.SELF,
    ) -> tuple[float, ...]:
        own = state.player(perspective)
        opponent = state.player(perspective.other)
        can_act = state.acting_player is perspective

        values = (
            min(state.turn_number / 30.0, 1.0),
            float(can_act),
            own.prizes_remaining / 6.0,
            opponent.prizes_remaining / 6.0,
            (opponent.prizes_remaining - own.prizes_remaining) / 6.0,
            min(len(own.hand) / 15.0, 1.0),
            min(len(opponent.hand) / 15.0, 1.0),
            len(own.deck) / 60.0,
            len(opponent.deck) / 60.0,
            len(own.discard) / 60.0,
            len(opponent.discard) / 60.0,
            len(own.bench) / 5.0,
            len(opponent.bench) / 5.0,
            float(own.active is not None),
            float(opponent.active is not None),
            _active_hp_fraction(own.active),
            _active_hp_fraction(opponent.active),
            _active_damage_pressure(own.active),
            _active_damage_pressure(opponent.active),
            min(_energy_in_play(own) / 12.0, 1.0),
            min(_energy_in_play(opponent) / 12.0, 1.0),
            _tools_in_play(own) / 6.0,
            _tools_in_play(opponent) / 6.0,
            float(can_act and not own.energy_attachment_used),
            float(can_act and not own.supporter_used),
            float(can_act and not own.retreat_used),
            float(state.stadium is not None),
            _pokemon_card_count(own, "Team Rocket's Mewtwo ex") / 2.0,
            _pokemon_card_count(own, "Team Rocket's Tarountula") / 4.0,
            _pokemon_card_count(own, "Team Rocket's Spidops") / 4.0,
            _pokemon_card_count(own, "Team Rocket's Articuno") / 2.0,
            _pokemon_card_count(own, "Team Rocket's Mimikyu") / 2.0,
            _pokemon_card_count(own, "Lillie's Clefairy ex"),
            _attached_energy_count(own, "Team Rocket's Energy") / 4.0,
            _attached_energy_count(own, "Basic Grass Energy") / 6.0,
            _attached_energy_count(own, "Basic Psychic Energy"),
            (5 - len(own.bench)) / 5.0,
        )
        if len(values) != self.feature_dim:
            raise RuntimeError("feature implementation does not match feature schema")
        return tuple(float(value) for value in values)

    def __call__(
        self,
        states: Iterable[GameState],
        *,
        perspective: PlayerId = PlayerId.SELF,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        rows = [self.tensorize_one(state, perspective=perspective) for state in states]
        if not rows:
            raise ValueError("at least one state is required")
        return torch.tensor(rows, dtype=dtype, device=device)
