"""Deterministic integration smoke demo for the Phase 1 substrate."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Callable

import torch

from .deck import CardCategory, load_default_deck
from .features import StateTensorizer
from .model import Phase1StrategicAgent
from .plan import PlanLatent
from .policy import CandidateBatch
from .state import (
    ActionKind,
    CardInstance,
    GameAction,
    GameState,
    MatchTrace,
    PlayerId,
    PlayerState,
    PokemonState,
    TurnPhase,
    TurnRecord,
)
from .strategy import STRATEGIC_AXES


def _expanded_self_deck() -> list[CardInstance]:
    deck = load_default_deck()
    instances: list[CardInstance] = []
    sequence = 0
    for entry in deck.entries:
        for _ in range(entry.count):
            instances.append(
                CardInstance(
                    instance_id=f"self-{sequence:03d}",
                    card_name=entry.name,
                    owner=PlayerId.SELF,
                )
            )
            sequence += 1
    return instances


def build_demo_state() -> GameState:
    """Build a synthetic but deck-integrity-preserving state for tests/demo."""

    available = _expanded_self_deck()

    def take(card_name: str) -> CardInstance:
        for index, card in enumerate(available):
            if card.card_name == card_name:
                return available.pop(index)
        raise RuntimeError(f"demo deck ran out of {card_name!r}")

    active = PokemonState(
        top_card=take("Team Rocket's Mewtwo ex"),
        max_hp=280,
        damage=60,
        attached_energy=(
            take("Team Rocket's Energy"),
            take("Basic Psychic Energy"),
        ),
        tool=take("Lucky Helmet"),
        effect_tags=("synthetic_demo_only",),
    )
    spidops = PokemonState(
        top_card=take("Team Rocket's Spidops"),
        max_hp=140,
        damage=20,
        evolution_stack=(take("Team Rocket's Tarountula"),),
        attached_energy=(take("Basic Grass Energy"),),
    )
    tarountula = PokemonState(
        top_card=take("Team Rocket's Tarountula"),
        max_hp=60,
    )
    hand = tuple(
        take(name)
        for name in (
            "Team Rocket's Ariana",
            "Ultra Ball",
            "Night Stretcher",
            "Basic Grass Energy",
        )
    )
    discard = tuple(
        take(name) for name in ("Lillie's Determination", "Bug Catching Set")
    )
    stadium = take("Prism Tower")
    prizes = tuple(available[:6])
    remaining_deck = tuple(available[6:])

    self_state = PlayerState(
        player_id=PlayerId.SELF,
        active=active,
        bench=(spidops, tarountula),
        hand=hand,
        deck=remaining_deck,
        prizes=prizes,
        discard=discard,
    )

    opponent_sequence = 0

    def opponent_card(
        *,
        name: str | None = None,
        revealed: bool = False,
    ) -> CardInstance:
        nonlocal opponent_sequence
        card = CardInstance(
            instance_id=f"opponent-{opponent_sequence:03d}",
            card_name=name,
            owner=PlayerId.OPPONENT,
            revealed=revealed,
        )
        opponent_sequence += 1
        return card

    opponent_active = PokemonState(
        top_card=opponent_card(name="Synthetic Opponent Pokemon", revealed=True),
        max_hp=200,
        damage=30,
        effect_tags=("synthetic_demo_only",),
    )
    opponent = PlayerState(
        player_id=PlayerId.OPPONENT,
        active=opponent_active,
        bench=(),
        hand=tuple(opponent_card() for _ in range(6)),
        deck=tuple(opponent_card() for _ in range(47)),
        prizes=tuple(opponent_card() for _ in range(6)),
    )

    return GameState(
        match_id="phase1-deterministic-demo",
        turn_number=3,
        acting_player=PlayerId.SELF,
        phase=TurnPhase.MAIN,
        players=(self_state, opponent),
        stadium=stadium,
    )


def _round(value: float) -> float:
    return round(float(value), 6)


def run_demo() -> dict[str, object]:
    """Exercise deck, state/history, plan, values, aggregator, and policy."""

    torch.manual_seed(20260726)
    torch.use_deterministic_algorithms(True)

    deck = load_default_deck()
    state = build_demo_state()
    state_roundtrip = GameState.from_json(state.to_json())

    next_state = replace(
        state,
        turn_number=4,
        acting_player=PlayerId.OPPONENT,
        phase=TurnPhase.DRAW,
    )
    end_turn = GameAction(
        sequence_index=0,
        actor=PlayerId.SELF,
        kind=ActionKind.END_TURN,
    )
    turn = TurnRecord(
        turn_number=state.turn_number,
        acting_player=PlayerId.SELF,
        state_before_digest=state.digest,
        actions=(end_turn,),
        state_after_digest=next_state.digest,
    )
    trace = MatchTrace(initial_state=state, current_state=state).append_turn(
        turn,
        next_state,
    )
    trace_roundtrip = MatchTrace.from_json(trace.to_json())

    tensorizer = StateTensorizer()
    features = tensorizer([state])
    agent = Phase1StrategicAgent()
    agent.eval()
    with torch.no_grad():
        evaluation = agent(features)
        updated = agent(features, evaluation.plan)
        plan_snapshot = evaluation.capture_plan(
            match_id=state.match_id,
            created_turn=state.turn_number,
            last_updated_turn=state.turn_number,
            update_count=1,
        )
        plan_roundtrip = PlanLatent.from_json(plan_snapshot.to_json())

        hidden = agent.tactical_policy.initial_hidden(
            evaluation.state_embedding,
            evaluation.plan,
        )
        candidate_features = torch.linspace(
            -1.0,
            1.0,
            steps=3 * agent.config.candidate_feature_dim,
        ).reshape(1, 3, agent.config.candidate_feature_dim)
        candidates = CandidateBatch(
            features=candidate_features,
            legal_mask=torch.tensor([[True, False, True]], dtype=torch.bool),
        )
        logits = agent.tactical_policy.score_next(hidden, candidates)
        next_hidden = agent.tactical_policy.advance(hidden, candidate_features[:, 0])

    category_counts = {
        category.value: deck.category_counts[category] for category in CardCategory
    }
    result: dict[str, object] = {
        "status": "ok",
        "deck": {
            "total_cards": deck.total_cards,
            "category_counts": category_counts,
            "fingerprint": deck.fingerprint,
        },
        "serialization": {
            "state_roundtrip": state_roundtrip == state,
            "trace_roundtrip": trace_roundtrip == trace,
            "plan_roundtrip": plan_roundtrip == plan_snapshot,
            "state_digest": state.digest,
        },
        "shapes": {
            "features": list(features.shape),
            "state_embedding": list(evaluation.state_embedding.shape),
            "plan": list(evaluation.plan.shape),
            "meta_value": list(evaluation.meta_value.shape),
            "meta_weights": list(evaluation.meta_weights.shape),
            "policy_logits": list(logits.shape),
            "policy_hidden": list(next_hidden.shape),
        },
        "strategy": {
            "head_order": [axis.value for axis in STRATEGIC_AXES],
            "head_values": {
                axis.value: _round(evaluation.head_values[axis][0, 0])
                for axis in STRATEGIC_AXES
            },
            "meta_value": _round(evaluation.meta_value[0, 0]),
            "meta_weight_sum": _round(evaluation.meta_weights[0].sum()),
            "plan_updates_across_turns": not torch.equal(evaluation.plan, updated.plan),
        },
        "policy": {
            "illegal_candidate_masked": bool(torch.isneginf(logits[0, 1]).item()),
            "legal_logits_finite": bool(
                torch.isfinite(logits[0, torch.tensor([0, 2])]).all().item()
            ),
        },
    }

    checks: tuple[tuple[str, Callable[[], bool]], ...] = (
        ("deck total", lambda: deck.total_cards == 60),
        ("state roundtrip", lambda: state_roundtrip == state),
        ("trace roundtrip", lambda: trace_roundtrip == trace),
        ("plan roundtrip", lambda: plan_roundtrip == plan_snapshot),
        ("meta weights", lambda: abs(float(evaluation.meta_weights.sum()) - 1.0) < 1e-6),
        ("policy mask", lambda: bool(torch.isneginf(logits[0, 1]).item())),
    )
    failed = [name for name, check in checks if not check()]
    if failed:
        raise RuntimeError(f"smoke demo failed checks: {failed}")
    return result


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
