#!/usr/bin/env python3
"""Collect paired short-horizon strategic targets at critical Rocket roots."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
from time import perf_counter, time
import random
import statistics
import sys
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import collect_target80_paired_critical_roots_v1 as v1  # noqa: E402
from collect_belief_rollout_teacher_n6 import _mean_and_se  # noqa: E402
from collect_live_truth_causal_action_values import (  # noqa: E402
    _valid_choice,
    option_summary,
    sha256,
    utility,
)
from collect_rocket_blueprint_n5_panel import tree_sha256  # noqa: E402


AXES = (
    "setup",
    "tempo",
    "resource_efficiency",
    "prize_control",
    "board_control",
    "optionality",
)

PLAN_WEIGHTS: dict[str, dict[str, float]] = {
    "baseline": {},
    "engine_resilience": {
        "spidops_engine": 1.25,
        "second_attacker": 1.25,
        "charging_fuel": 0.75,
        "bench_discipline": 0.50,
        "fuel_preservation": 0.50,
        "attack_tempo": 0.25,
    },
    "tempo_conversion": {
        "attack_tempo": 1.25,
        "fuel_preservation": 0.75,
        "second_attacker": 0.50,
        "tool_assignment": 0.50,
        "bench_discipline": 0.50,
    },
}
PLAN_NAMES = tuple(PLAN_WEIGHTS)
PLAN_TIE_ORDER = {name: index for index, name in enumerate(PLAN_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "team_rocket_recovery"
        / "candidates"
        / "scale_neural_resolver_safe_tie_v3",
    )
    parser.add_argument(
        "--own-candidate",
        type=Path,
        default=PROJECT_ROOT
        / "work"
        / "target80_candidates"
        / "mimikyu_maxbelt_neural_control_v1",
    )
    parser.add_argument(
        "--opponent",
        type=Path,
        default=PROJECT_ROOT
        / "work"
        / "arena_bundles"
        / "AshV178MarnieFullBest_20260801_extracted",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", choices=("first", "second"), required=True)
    parser.add_argument("--capture-roots-per-game", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--minimum-plan-action-support", type=float, default=0.01)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--coin-rollouts", type=int, default=2)
    parser.add_argument("--particle-offset", type=int, default=0)
    parser.add_argument("--earliest-turn", type=int, default=2)
    parser.add_argument("--latest-turn", type=int, default=8)
    parser.add_argument("--horizon-turns", type=int, default=2)
    parser.add_argument("--max-source-attempts", type=int, default=200)
    parser.add_argument("--max-live-decisions", type=int, default=2_000)
    parser.add_argument("--max-rollout-decisions", type=int, default=250)
    parser.add_argument("--root-budget-seconds", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "roots",
        "capture_roots_per_game",
        "candidates",
        "particles",
        "coin_rollouts",
        "horizon_turns",
        "max_source_attempts",
        "max_live_decisions",
        "max_rollout_decisions",
        "threads",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.horizon_turns % 2:
        raise ValueError("--horizon-turns must be an even number of alternating turns")
    if not 0.0 < args.minimum_plan_action_support < 1.0:
        raise ValueError("--minimum-plan-action-support must be in (0, 1)")
    if args.particle_offset < 0 or args.root_budget_seconds <= 0.0:
        raise ValueError("invalid particle offset or root budget")
    if args.earliest_turn < 0 or args.latest_turn < args.earliest_turn:
        raise ValueError("invalid turn window")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite V2 artifact: {args.output}")


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return min(high, max(low, float(value)))


def _energy_units(pokemon: Mapping[str, Any]) -> int:
    cards = [
        int(card.get("id") or 0)
        for card in pokemon.get("energyCards") or ()
        if isinstance(card, Mapping)
    ]
    if cards:
        return sum(2 if card_id == 15 else 1 for card_id in cards)
    return len(pokemon.get("energies") or ())


def _active_damage_fraction(player: Mapping[str, Any]) -> float:
    active = [card for card in player.get("active") or () if isinstance(card, Mapping)]
    if not active:
        return 1.0
    hp = max(0.0, float(active[0].get("hp") or 0.0))
    maximum = max(1.0, float(active[0].get("maxHp") or hp or 1.0))
    return _clip(1.0 - hp / maximum, 0.0, 1.0)


def _ready_cards(player: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    from persistent_plan_runtime import ATTACKERS, ROCKET_POKEMON, _energy_card_ids, _ready

    board = v1._board(player)
    rocket_count = sum(int(card.get("id") or 0) in ROCKET_POKEMON for card in board)
    return [
        card
        for card in board
        if int(card.get("id") or 0) in ATTACKERS
        and _ready(int(card.get("id") or 0), _energy_card_ids(card), rocket_count)
    ]


def strategic_axes(observation: Mapping[str, Any], root_player: int) -> dict[str, float]:
    """Return six bounded public strategic values from the root player's view."""

    current = observation.get("current") or {}
    players = current.get("players") or ()
    if len(players) != 2:
        raise ValueError("milestone observation lacks two players")
    own = players[root_player] or {}
    opponent = players[1 - root_player] or {}
    own_board = v1._board(own)
    opponent_board = v1._board(opponent)
    ready = _ready_cards(own)
    ready_serials = {int(card.get("serial") or -1) for card in ready}
    active_serials = {
        int(card.get("serial") or -2)
        for card in own.get("active") or ()
        if isinstance(card, Mapping)
    }
    active_ready = bool(ready_serials & active_serials)
    rocket_count = sum(
        int(card.get("id") or 0) in {400, 401, 414, 431, 463}
        for card in own_board
    )
    spidops_count = sum(int(card.get("id") or 0) == v1.SPIDOPS for card in own_board)
    mewtwo_on_board = any(int(card.get("id") or 0) == v1.MEWTWO for card in own_board)
    mewtwo_with_backup = bool(
        any(int(card.get("id") or 0) == v1.MEWTWO for card in ready)
        and any(int(card.get("id") or 0) != v1.MEWTWO for card in ready)
    )
    total_energy = sum(_energy_units(card) for card in own_board)
    ready_energy = sum(_energy_units(card) for card in ready)
    useful_ratio = ready_energy / total_energy if total_energy else 0.0
    own_damage = _active_damage_fraction(own)
    opponent_damage = _active_damage_fraction(opponent)
    own_remaining = len(own.get("prize") or ())
    opponent_remaining = len(opponent.get("prize") or ())
    prize_advantage = (opponent_remaining - own_remaining) / 6.0
    hand_count = float(own.get("handCount") or len(own.get("hand") or ()))
    deck_count = float(own.get("deckCount") or 0)
    open_bench = max(0.0, float(own.get("benchMax") or 5) - len(own.get("bench") or ()))

    setup_01 = (
        0.30 * min(1.0, rocket_count / 5.0)
        + 0.30 * min(1.0, spidops_count / 3.0)
        + 0.20 * float(mewtwo_on_board)
        + 0.20 * min(1.0, len(ready) / 2.0)
    )
    tempo_01 = _clip(
        0.45 * min(1.0, len(ready) / 2.0)
        + 0.25 * float(active_ready)
        + 0.20 * opponent_damage
        + 0.10 * (1.0 - own_damage),
        0.0,
        1.0,
    )
    resource_01 = _clip(
        0.45 * useful_ratio
        + 0.30 * min(1.0, total_energy / 5.0)
        + 0.25 * min(1.0, len(ready) / 2.0),
        0.0,
        1.0,
    )
    prize_value = _clip(0.70 * prize_advantage + 0.30 * (opponent_damage - own_damage))
    board_01 = _clip(
        0.55 * min(1.0, len(own_board) / 6.0)
        + 0.30 * min(1.0, len(ready) / 2.0)
        + 0.15 * (1.0 - min(1.0, len(opponent_board) / 6.0)),
        0.0,
        1.0,
    )
    optionality_01 = _clip(
        0.30 * min(1.0, open_bench / 5.0)
        + 0.25 * min(1.0, hand_count / 10.0)
        + 0.20 * min(1.0, deck_count / 40.0)
        + 0.25 * float(mewtwo_with_backup),
        0.0,
        1.0,
    )
    return {
        "setup": 2.0 * setup_01 - 1.0,
        "tempo": 2.0 * tempo_01 - 1.0,
        "resource_efficiency": 2.0 * resource_01 - 1.0,
        "prize_control": prize_value,
        "board_control": 2.0 * board_01 - 1.0,
        "optionality": 2.0 * optionality_01 - 1.0,
    }


def meta_weights(root: Mapping[str, Any], critical: Mapping[str, Any]) -> dict[str, float]:
    """Freeze one state-conditioned strategic mixture for all root branches."""

    flags = critical["flags"]
    if flags["prize_threshold"]:
        values = {
            "setup": 0.10,
            "tempo": 0.25,
            "resource_efficiency": 0.10,
            "prize_control": 0.30,
            "board_control": 0.15,
            "optionality": 0.10,
        }
    elif flags["mewtwo_activation"]:
        values = {
            "setup": 0.25,
            "tempo": 0.15,
            "resource_efficiency": 0.25,
            "prize_control": 0.10,
            "board_control": 0.15,
            "optionality": 0.10,
        }
    elif flags["missing_second_attacker"]:
        values = {
            "setup": 0.25,
            "tempo": 0.20,
            "resource_efficiency": 0.20,
            "prize_control": 0.10,
            "board_control": 0.15,
            "optionality": 0.10,
        }
    else:
        values = {
            "setup": 0.20,
            "tempo": 0.20,
            "resource_efficiency": 0.15,
            "prize_control": 0.20,
            "board_control": 0.15,
            "optionality": 0.10,
        }
    if abs(sum(values.values()) - 1.0) > 1e-9:
        raise RuntimeError("meta weights do not sum to one")
    return values


def aggregate_axes(axes: Mapping[str, float], weights: Mapping[str, float]) -> float:
    if set(axes) != set(AXES) or set(weights) != set(AXES):
        raise ValueError("strategic axis mismatch")
    return sum(float(weights[name]) * float(axes[name]) for name in AXES)


def _best_plan(
    paired: Sequence[Mapping[str, Any]],
    plans: Sequence[str],
    *,
    particles: set[int] | None = None,
    coin: int | None = None,
) -> str | None:
    selected = [
        row
        for row in paired
        if (particles is None or int(row["particle"]) in particles)
        and (coin is None or int(row["coin_rollout"]) == coin)
    ]
    if not selected:
        return None
    means = {
        name: statistics.fmean(float(row["values"][name]) for row in selected)
        for name in plans
    }
    return max(plans, key=lambda name: (means[name], -PLAN_TIE_ORDER[name]))


def choose_milestone_root(
    runtime: "MilestoneRuntime",
    roots: Sequence[Mapping[str, Any]],
    *,
    minimum_plan_action_support: float,
) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, Any]] | None:
    ranked = []
    for root in roots:
        diagnostics = v1.policy_diagnostics(
            runtime,
            root,
            minimum_alternative_support=minimum_plan_action_support,
        )
        plan_actions = runtime.plan_root_actions(root)
        unique_actions = sorted({tuple(action) for action in plan_actions.values()})
        if len(unique_actions) < 2:
            continue
        probabilities = diagnostics["policy_probabilities"]
        if any(
            name != "baseline"
            and float(probabilities[action[0]]) < minimum_plan_action_support
            for name, action in plan_actions.items()
        ):
            continue
        diagnostics = dict(diagnostics)
        diagnostics["plan_actions"] = plan_actions
        diagnostics["candidate_indices"] = sorted({int(action[0]) for action in plan_actions.values()})
        critical = v1.criticality(root, diagnostics)
        flags = critical["flags"]
        eligible = bool(
            flags["low_policy_margin"]
            or flags["attack_setup_tradeoff"]
            or flags["mewtwo_activation"]
            or (flags["missing_second_attacker"] and int(root["turn"]) >= 3)
        )
        if not eligible:
            continue
        margin = float(diagnostics["policy_margin"])
        score = (
            8.0 * float(flags["attack_setup_tradeoff"])
            + 6.0 * float(flags["mewtwo_activation"])
            + 4.0 * float(flags["missing_second_attacker"])
            + 3.0 * float(flags["spidops_engine"])
            + 2.0 * float(flags["low_policy_margin"])
            - 0.15 * abs(float(root["turn"]) - 5.0)
            - min(3.0, margin if math.isfinite(margin) else 3.0)
        )
        critical = dict(critical)
        critical["milestone_priority_score"] = score
        critical["distinct_plan_root_actions"] = len(unique_actions)
        rank = (score, -abs(int(root["turn"]) - 5), int(root["step"]), str(root["observation_fingerprint"]))
        ranked.append((rank, root, diagnostics, critical))
    if not ranked:
        return None
    _, root, diagnostics, critical = max(ranked, key=lambda item: item[0])
    return root, diagnostics, critical


class MilestoneRuntime(v1.CriticalRootRuntime):
    def branch_agents(
        self,
        beliefs: tuple[Any, Any],
        plan_name: str,
    ) -> tuple[Any, Any]:
        if plan_name not in PLAN_WEIGHTS:
            raise KeyError(f"unknown strategic plan {plan_name}")
        baseline_own, opponent = super().new_agents(beliefs)
        if plan_name == "baseline":
            return baseline_own, opponent
        from persistent_plan_runtime import PersistentPlanAgent

        plan = PersistentPlanAgent(
            self.own_model,
            PLAN_WEIGHTS[plan_name],
            plan_name=plan_name,
            deck_ids=self.own_deck,
            opponent_deck_priors=self.deck_library,
            forbid_end_when_attack=True,
            max_abs_bias=1.5,
        )
        plan.belief = copy.deepcopy(beliefs[0])
        plan.blueprint.belief = plan.belief
        guarded = v1.GuardedAgent(plan, self.guard_module)
        return guarded, opponent

    def plan_root_actions(self, root: Mapping[str, Any]) -> dict[str, list[int]]:
        actions: dict[str, list[int]] = {}
        for plan_name in PLAN_NAMES:
            own, _ = self.branch_agents(root["_beliefs"], plan_name)
            action = [int(index) for index in own(root["observation"])]
            if not _valid_choice(root["observation"], action) or len(action) != 1:
                raise RuntimeError(f"{plan_name} produced an invalid root action: {action}")
            actions[plan_name] = action
        baseline = [int(index) for index in root["baseline_action"]]
        if actions["baseline"] != baseline:
            raise RuntimeError(
                f"baseline plan did not reproduce source action: {actions['baseline']} != {baseline}"
            )
        return actions

    def _horizon_rollout(
        self,
        branch: Any,
        *,
        plan_name: str,
        root_player: int,
        root_turn: int,
        horizon_turns: int,
        beliefs: tuple[Any, Any],
        weights: Mapping[str, float],
        max_decisions: int,
        deadline: float,
        coin_seed: int,
    ) -> dict[str, Any]:
        actors_base = self.branch_agents(beliefs, plan_name)
        actors = actors_base if root_player == 0 else (actors_base[1], actors_base[0])
        coin_rng = random.Random(coin_seed)
        coin_flips = 0
        for step in range(max_decisions):
            raw = asdict(branch.observation)
            current = raw.get("current") or {}
            result = int(current.get("result") if current.get("result") is not None else -1)
            if result != -1:
                terminal_utility = utility(result, root_player)
                axes = strategic_axes(raw, root_player)
                return {
                    "complete": True,
                    "terminal": True,
                    "terminal_utility": terminal_utility,
                    "value": 4.0 * terminal_utility,
                    "axes": axes,
                    "depth": step,
                    "horizon_turn": int(current.get("turn") or root_turn),
                    "coin_flips": coin_flips,
                }
            turn = int(current.get("turn") or 0)
            actor = int(current.get("yourIndex") or 0)
            if turn >= root_turn + horizon_turns and actor == root_player:
                axes = strategic_axes(raw, root_player)
                return {
                    "complete": True,
                    "terminal": False,
                    "terminal_utility": None,
                    "value": aggregate_axes(axes, weights),
                    "axes": axes,
                    "depth": step,
                    "horizon_turn": turn,
                    "coin_flips": coin_flips,
                }
            if perf_counter() >= deadline or raw.get("select") is None:
                break
            select = raw.get("select") or {}
            if int(select.get("context") or -1) == 46:
                want_type = 1 if coin_rng.getrandbits(1) else 2
                matches = [
                    index
                    for index, option in enumerate(select.get("option") or ())
                    if isinstance(option, Mapping) and int(option.get("type") or -1) == want_type
                ]
                if len(matches) != 1:
                    raise RuntimeError("manual coin prompt did not expose one head/tail option")
                action = [matches[0]]
                coin_flips += 1
            else:
                action = actors[actor](raw)
            if not _valid_choice(raw, action):
                raise RuntimeError(f"milestone policy returned illegal action: {action}")
            branch = self.search_step(branch.searchId, action)
        return {
            "complete": False,
            "terminal": False,
            "terminal_utility": None,
            "value": None,
            "axes": None,
            "depth": max_decisions,
            "horizon_turn": None,
            "coin_flips": coin_flips,
        }

    def evaluate_milestone_root(
        self,
        root: Mapping[str, Any],
        *,
        critical: Mapping[str, Any],
        particles: int,
        particle_offset: int,
        coin_rollouts: int,
        horizon_turns: int,
        max_rollout_decisions: int,
        budget_seconds: float,
        seed: int,
    ) -> dict[str, Any]:
        deadline = perf_counter() + budget_seconds
        plans = PLAN_NAMES
        baseline = "baseline"
        plan_actions = {
            name: [int(index) for index in action]
            for name, action in root["plan_actions"].items()
        }
        weights = meta_weights(root, critical)
        paired: list[dict[str, Any]] = []
        particle_manifests: list[dict[str, Any]] = []
        errors: list[str] = []
        attempted_pairs = 0
        for local_particle in range(particles):
            particle = particle_offset + local_particle
            if perf_counter() >= deadline:
                break
            allocation_seed = seed + 1_000_003 * particle
            try:
                completion = self._completion(
                    root,
                    particle=particle,
                    total_particles=particles,
                    allocation_seed=allocation_seed,
                )
            except Exception as error:
                errors.append(
                    f"particle={particle}: completion {type(error).__name__}: {str(error)[:180]}"
                )
                continue
            particle_manifests.append(completion.manifest)
            search_root = None
            try:
                search_root = self.search_begin(
                    self.to_observation_class(root["observation"]),
                    **completion.search_arguments(),
                )
                for coin_rollout in range(coin_rollouts):
                    if perf_counter() >= deadline:
                        break
                    attempted_pairs += 1
                    values: dict[str, float] = {}
                    axes: dict[str, dict[str, float]] = {}
                    terminals: dict[str, bool] = {}
                    terminal_utilities: dict[str, float | None] = {}
                    depths: dict[str, int] = {}
                    horizon_reached: dict[str, int] = {}
                    coin_flips: dict[str, int] = {}
                    order = list(plans)
                    random.Random(seed + 10_007 * particle + coin_rollout).shuffle(order)
                    coin_seed = seed + 65_537 * particle + 977 * coin_rollout
                    for plan_name in order:
                        if perf_counter() >= deadline:
                            break
                        try:
                            child = self.search_step(search_root.searchId, plan_actions[plan_name])
                            outcome = self._horizon_rollout(
                                child,
                                plan_name=plan_name,
                                root_player=int(root["_own_seat"]),
                                root_turn=int(root["turn"]),
                                horizon_turns=horizon_turns,
                                beliefs=root["_beliefs"],
                                weights=weights,
                                max_decisions=max_rollout_decisions,
                                deadline=deadline,
                                coin_seed=coin_seed,
                            )
                            if outcome["complete"]:
                                key = plan_name
                                values[key] = float(outcome["value"])
                                axes[key] = {name: float(outcome["axes"][name]) for name in AXES}
                                terminals[key] = bool(outcome["terminal"])
                                terminal_utilities[key] = outcome["terminal_utility"]
                                depths[key] = int(outcome["depth"])
                                horizon_reached[key] = int(outcome["horizon_turn"])
                                coin_flips[key] = int(outcome["coin_flips"])
                        except Exception as error:
                            errors.append(
                                f"particle={particle} coin={coin_rollout} plan={plan_name}: "
                                f"{type(error).__name__}: {str(error)[:180]}"
                            )
                    if len(values) == len(plans):
                        paired.append(
                            {
                                "particle": particle,
                                "coin_rollout": coin_rollout,
                                "values": values,
                                "axes": axes,
                                "terminals": terminals,
                                "terminal_utilities": terminal_utilities,
                                "depths": depths,
                                "horizon_turns": horizon_reached,
                                "coin_flips": coin_flips,
                            }
                        )
            finally:
                if search_root is not None:
                    try:
                        self.search_end()
                    except Exception as error:
                        errors.append(
                            f"particle={particle}: search_end {type(error).__name__}: {str(error)[:180]}"
                        )

        samples = {
            name: [float(row["values"][name]) for row in paired]
            for name in plans
        }
        means = {name: _mean_and_se(values)[0] for name, values in samples.items()}
        standard_errors = {name: _mean_and_se(values)[1] for name, values in samples.items()}
        best = _best_plan(paired, plans) if paired else baseline
        deltas = {
            name: [
                float(row["values"][name]) - float(row["values"][baseline])
                for row in paired
            ]
            for name in plans
        }
        particle_ids = list(range(particle_offset, particle_offset + particles))
        midpoint = len(particle_ids) // 2
        return {
            "baseline": baseline,
            "best": best,
            "means": means,
            "standard_errors": standard_errors,
            "paired_deltas": deltas,
            "half_best": [
                _best_plan(paired, plans, particles=set(particle_ids[:midpoint])),
                _best_plan(paired, plans, particles=set(particle_ids[midpoint:])),
            ],
            "paired_samples": paired,
            "meta_weights": weights,
            "complete_pairs": len(paired),
            "attempted_pairs": attempted_pairs,
            "fully_complete": len(paired) == particles * coin_rollouts,
            "particle_manifests": particle_manifests,
            "errors": errors,
        }


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    runtime_path, own, opponent = v1._validate_frozen_inputs(args)
    runtime = MilestoneRuntime(runtime_path, own, opponent, args.candidates)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    retained = attempts = 0
    started = time()
    with output.open("x", encoding="utf-8") as sink:
        while retained < args.roots and attempts < args.max_source_attempts:
            attempts += 1
            capture = runtime.capture_match(
                seat_mode=args.seat,
                accepted_game_index=retained + 1,
                roots_per_game=args.capture_roots_per_game,
                earliest_turn=args.earliest_turn,
                latest_turn=args.latest_turn,
                max_decisions=args.max_live_decisions,
                common_random_coins=True,
                capture_truth=False,
            )
            if capture.get("seat_skipped"):
                counters["seat_skips"] += 1
                continue
            counters["source_games"] += 1
            selected = choose_milestone_root(
                runtime,
                capture["roots"],
                minimum_plan_action_support=args.minimum_plan_action_support,
            )
            if selected is None:
                counters["source_games_without_eligible_root"] += 1
                continue
            root, diagnostics, critical = selected
            root = dict(root)
            plan_actions = {
                name: [int(index) for index in action]
                for name, action in diagnostics["plan_actions"].items()
            }
            root["plan_actions"] = plan_actions
            options = (root["observation"].get("select") or {}).get("option") or ()
            root["plan_root_options"] = {
                name: option_summary(options[action[0]], root["observation"])
                for name, action in plan_actions.items()
            }
            evaluation = runtime.evaluate_milestone_root(
                root,
                critical=critical,
                particles=args.particles,
                particle_offset=args.particle_offset,
                coin_rollouts=args.coin_rollouts,
                horizon_turns=args.horizon_turns,
                max_rollout_decisions=args.max_rollout_decisions,
                budget_seconds=args.root_budget_seconds,
                seed=args.seed + 100_003 * (retained + 1),
            )
            retained += 1
            baseline_utility = utility(int(capture["result"]), int(capture["own_seat"]))
            counters["retained_roots"] += 1
            counters["fully_complete_roots"] += int(evaluation["fully_complete"])
            counters["complete_pairs"] += int(evaluation["complete_pairs"])
            counters["attempted_pairs"] += int(evaluation["attempted_pairs"])
            counters["rollout_errors"] += len(evaluation["errors"])
            counters["guarded_roots"] += int(diagnostics["guard_name"] is not None)
            for name, active in critical["flags"].items():
                counters[f"critical_{name}"] += int(active)
            row = {
                "schema": "target80-paired-milestone-root/v2",
                "protocol": "TARGET80_PAIRED_MILESTONE_ROOT_V2_PROTOCOL.md",
                "root_id": retained,
                "source_attempt": attempts,
                "opponent": opponent.name,
                "own_seat": capture["own_seat"],
                "started_first": capture["started_first"],
                "first_player": capture["first_player"],
                "baseline_source_result": capture["result"],
                "baseline_source_utility": baseline_utility,
                "baseline_source_decisions": capture["decisions"],
                "step": root["step"],
                "turn": root["turn"],
                "turn_action_count": root["turn_action_count"],
                "baseline_action": root["baseline_action"],
                "plan_names": list(PLAN_NAMES),
                "plan_weights": PLAN_WEIGHTS,
                "plan_actions": plan_actions,
                "plan_root_options": root["plan_root_options"],
                "observation_fingerprint": root["observation_fingerprint"],
                "observation": root["observation"],
                "policy_feature": diagnostics["policy_feature"],
                "policy_logits": diagnostics["policy_logits"],
                "policy_probabilities": diagnostics["policy_probabilities"],
                "policy_argmax": diagnostics["policy_argmax"],
                "guard_name": diagnostics["guard_name"],
                "guarded_baseline_reproduced": diagnostics["guarded_baseline_reproduced"],
                "policy_margin": diagnostics["policy_margin"],
                "criticality": critical,
                **evaluation,
            }
            sink.write(json.dumps(row, separators=(",", ":")) + "\n")
            sink.flush()
            print(
                json.dumps(
                    {
                        "root": retained,
                        "of": args.roots,
                        "source_attempt": attempts,
                        "turn": root["turn"],
                        "complete_pairs": evaluation["complete_pairs"],
                        "fully_complete": evaluation["fully_complete"],
                        "errors": len(evaluation["errors"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if retained != args.roots:
        raise RuntimeError(f"retained only {retained}/{args.roots} roots in {attempts} attempts")
    manifest = {
        "schema": "target80-paired-milestone-root-manifest/v2",
        "protocol": "TARGET80_PAIRED_MILESTONE_ROOT_V2_PROTOCOL.md",
        "method": "paired public-belief fixed-horizon six-axis strategic milestones",
        "genuine_official_engine_source_matches": True,
        "genuine_official_engine_descendant_transitions": True,
        "source_policy_exact_packaged_public_guards": True,
        "visualizer_or_realized_hidden_truth_read": False,
        "hidden_completion_distribution_official": False,
        "raw_hidden_allocations_serialized": False,
        "actor_information": "normal redacted observation plus persistent public/own-visible belief",
        "public_information_set_expected_value": False,
        "equilibrium_or_safety_certificate": False,
        "common_random_coin_sequences": True,
        "leaf_value_substitution": False,
        "sampled_rollout_is_final_strategy": False,
        "deployment_authority": False,
        "strategic_axes": list(AXES),
        "strategic_plans": PLAN_WEIGHTS,
        "plan_max_abs_logit_bias": 1.5,
        "plan_persists_through_horizon": True,
        "terminal_value_dominates_dense_score": True,
        "runtime": str(runtime_path),
        "own_candidate": str(own),
        "opponent": str(opponent),
        "own_tree_sha256": tree_sha256(own),
        "own_deck_sha256": sha256(own / "deck.csv"),
        "own_model_sha256": sha256(own / "model.pt"),
        "own_main_sha256": sha256(own / "main.py"),
        "opponent_tree_sha256": tree_sha256(opponent),
        "opponent_deck_sha256": sha256(opponent / "deck.csv"),
        "opponent_model_sha256": sha256(opponent / "model.pt"),
        "opponent_main_sha256": sha256(opponent / "main.py"),
        "collector_sha256": sha256(Path(__file__).resolve()),
        "v1_dependency_sha256": sha256(Path(v1.__file__).resolve()),
        "output": str(output),
        "output_sha256": sha256(output),
        "attempted_matches": attempts,
        "counts": dict(counters),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "elapsed_seconds": time() - started,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
