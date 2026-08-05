#!/usr/bin/env python3
"""Collect paired official-engine continuations at critical Wobbuffet-free roots.

The source match and every descendant transition use the official ``cg``
engine.  Hidden zones are sampled only from the acting player's redacted
observation and frozen deck priors.  This collector is an offline teacher-data
pilot, not a deployed search policy, an equilibrium solver, or a safety proof.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
from time import time
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from collect_belief_rollout_teacher_n6 import BeliefRolloutRuntime  # noqa: E402
from collect_live_truth_causal_action_values import (  # noqa: E402
    fingerprint,
    option_summary,
    sha256,
    utility,
)
from collect_rocket_blueprint_n5_panel import tree_sha256  # noqa: E402
from rocketbot.policy_feature_snapshot import serialize_policy_feature  # noqa: E402


OWN_TREE_SHA256 = "30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f"
OWN_DECK_SHA256 = "4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea"
OWN_MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
OWN_MAIN_SHA256 = "80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6"

MARNIE_TREE_SHA256 = "49bb3353ab42fded18e32e38f154ad1a001d60c5852383045ff4f437be36efe1"
MARNIE_DECK_SHA256 = "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d"
MARNIE_MAIN_SHA256 = "0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b"

WOBBUFFET = 432
MEWTWO = 431
SPIDOPS = 401
PLAY = 7
ATTACH = 8
EVOLVE = 9
ABILITY = 10
RETREAT = 12
ATTACK = 13
END = 14
SETUP_TYPES = frozenset({PLAY, ATTACH, EVOLVE, ABILITY, RETREAT})


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
    parser.add_argument("--minimum-alternative-support", type=float, default=0.02)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--coin-rollouts", type=int, default=2)
    parser.add_argument("--particle-offset", type=int, default=0)
    parser.add_argument("--earliest-turn", type=int, default=2)
    parser.add_argument("--latest-turn", type=int, default=10)
    parser.add_argument("--max-source-attempts", type=int, default=200)
    parser.add_argument("--max-live-decisions", type=int, default=2_000)
    parser.add_argument("--max-rollout-decisions", type=int, default=600)
    parser.add_argument("--root-budget-seconds", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "roots",
        "capture_roots_per_game",
        "candidates",
        "particles",
        "coin_rollouts",
        "max_source_attempts",
        "max_live_decisions",
        "max_rollout_decisions",
        "threads",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.minimum_alternative_support < 1.0:
        raise ValueError("--minimum-alternative-support must be in (0, 1)")
    if args.particle_offset < 0 or args.root_budget_seconds <= 0.0:
        raise ValueError("invalid particle offset or root budget")
    if args.earliest_turn < 0 or args.latest_turn < args.earliest_turn:
        raise ValueError("invalid turn window")
    manifest = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest.exists():
        raise FileExistsError(f"refusing to overwrite pilot artifact: {args.output}")


def _load_guard_module(candidate: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "target80_paired_critical_own_main",
        candidate / "main.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load frozen candidate main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("_early_spidops_search_override", "_spidops_bangle_lethal_override"):
        if not callable(getattr(module, name, None)):
            raise AttributeError(f"frozen candidate is missing guard {name}")
    return module


class GuardedAgent:
    """Apply the exact packaged public guards before the frozen NNAgent."""

    def __init__(self, base: Any, guard_module: Any):
        self._base = base
        self._guards = (
            guard_module._early_spidops_search_override,
            guard_module._spidops_bangle_lethal_override,
        )

    @property
    def belief(self) -> Any:
        return self._base.belief

    @belief.setter
    def belief(self, value: Any) -> None:
        self._base.belief = value

    def guard_action(self, observation: Mapping[str, Any]) -> tuple[list[int] | None, str | None]:
        for guard in self._guards:
            action = guard(observation)
            if action is not None:
                return [int(index) for index in action], str(guard.__name__)
        return None, None

    def __call__(self, observation: Mapping[str, Any]) -> list[int]:
        action, _ = self.guard_action(observation)
        return action if action is not None else self._base(observation)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class CriticalRootRuntime(BeliefRolloutRuntime):
    """Official engine runtime with the exact current Wobbuffet-free guards."""

    def __init__(self, runtime: Path, own: Path, opponent: Path, candidates: int):
        super().__init__(runtime, own, opponent, candidates)
        self.guard_module = _load_guard_module(self.own_path)

    def new_agents(self, beliefs: tuple[Any, Any] | None = None) -> tuple[Any, Any]:
        own, opponent = super().new_agents(beliefs)
        return GuardedAgent(own, self.guard_module), opponent

    def guard_action(self, observation: Mapping[str, Any]) -> tuple[list[int] | None, str | None]:
        probe = GuardedAgent.__new__(GuardedAgent)
        probe._guards = (
            self.guard_module._early_spidops_search_override,
            self.guard_module._spidops_bangle_lethal_override,
        )
        return GuardedAgent.guard_action(probe, observation)


def _board(player: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        pokemon
        for zone in ("active", "bench")
        for pokemon in player.get(zone) or ()
        if isinstance(pokemon, Mapping)
    ]


def _ready_rocket_attackers(player: Mapping[str, Any]) -> tuple[int, bool]:
    from persistent_plan_runtime import ATTACKERS, MEWTWO as PLAN_MEWTWO
    from persistent_plan_runtime import ROCKET_POKEMON, _energy_card_ids, _ready

    board = _board(player)
    rocket_count = sum(int(card.get("id") or 0) in ROCKET_POKEMON for card in board)
    ready = [
        card
        for card in board
        if int(card.get("id") or 0) in ATTACKERS
        and _ready(
            int(card.get("id") or 0),
            _energy_card_ids(card),
            rocket_count,
        )
    ]
    mewtwo_ready = any(int(card.get("id") or 0) == PLAN_MEWTWO for card in ready)
    return len(ready), mewtwo_ready


@torch.no_grad()
def policy_diagnostics(
    runtime: CriticalRootRuntime,
    root: Mapping[str, Any],
    *,
    minimum_alternative_support: float,
) -> dict[str, Any]:
    """Reproduce the exact guarded action and expose actor-visible uncertainty."""

    from soldier import features
    from soldier.net import attack_safe_logits, feat_to_tensors

    belief = copy.deepcopy(root["_beliefs"][0])
    feature = features.extract(
        root["observation"],
        belief=belief,
        slot_link=bool(getattr(runtime.own_model, "slot_link", False)),
    )
    tensors = feat_to_tensors(feature, "cpu")
    logits, _ = runtime.own_model(tensors)
    options = (root["observation"].get("select") or {}).get("option") or ()
    if any(int(option.get("type") or -1) == ATTACK for option in options):
        logits = attack_safe_logits(logits, options)
    probabilities = torch.softmax(logits, dim=-1)
    direct = int(torch.argmax(logits))
    guard_action, guard_name = runtime.guard_action(root["observation"])
    expected = int(guard_action[0]) if guard_action is not None else direct
    baseline = int(root["baseline_action"][0])
    if expected != baseline:
        raise RuntimeError(
            f"guarded actor-feature parity failed: expected {expected}, captured {baseline}"
        )

    original = [int(index) for index in root["candidate_indices"]]
    retained = [baseline]
    for index in original:
        if index == baseline:
            continue
        if float(probabilities[index]) >= minimum_alternative_support:
            retained.append(index)
    retained = retained[: runtime.candidates]
    finite = sorted(
        (float(value), index)
        for index, value in enumerate(logits.tolist())
        if math.isfinite(float(value))
    )
    policy_margin = math.inf
    if len(finite) >= 2:
        policy_margin = float(finite[-1][0] - finite[-2][0])
    return {
        "policy_feature": serialize_policy_feature(feature),
        "policy_logits": [float(value) for value in logits.tolist()],
        "policy_probabilities": [float(value) for value in probabilities.tolist()],
        "policy_argmax": direct,
        "guard_name": guard_name,
        "guarded_baseline": baseline,
        "guarded_baseline_reproduced": True,
        "candidate_indices": retained,
        "policy_margin": policy_margin,
    }


def criticality(
    root: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    observation = root["observation"]
    current = observation.get("current") or {}
    players = current.get("players") or ()
    own_seat = int(root["_own_seat"])
    if len(players) != 2:
        raise ValueError("critical root lacks two public player states")
    own = players[own_seat] or {}
    opponent = players[1 - own_seat] or {}
    own_board = _board(own)
    ready_count, mewtwo_ready = _ready_rocket_attackers(own)
    candidate_indices = [int(index) for index in diagnostics["candidate_indices"]]
    options = (observation.get("select") or {}).get("option") or ()
    option_types = {int(options[index].get("type") or 0) for index in candidate_indices}
    spidops_count = sum(int(card.get("id") or 0) == SPIDOPS for card in own_board)
    mewtwo_on_board = any(int(card.get("id") or 0) == MEWTWO for card in own_board)
    own_prizes = len(own.get("prize") or ())
    opponent_prizes = len(opponent.get("prize") or ())
    margin = float(diagnostics["policy_margin"])
    flags = {
        "low_policy_margin": margin <= 1.0,
        "attack_setup_tradeoff": ATTACK in option_types and bool(option_types & SETUP_TYPES),
        "missing_second_attacker": ready_count < 2 and bool(option_types & {PLAY, ATTACH, EVOLVE}),
        "mewtwo_activation": mewtwo_on_board and not mewtwo_ready and bool(option_types & {PLAY, ATTACH}),
        "spidops_engine": spidops_count < 3 and bool(option_types & {PLAY, EVOLVE}),
        "prize_threshold": min(own_prizes, opponent_prizes) <= 3,
    }
    eligible = bool(
        flags["low_policy_margin"]
        or flags["attack_setup_tradeoff"]
        or flags["mewtwo_activation"]
        or flags["prize_threshold"]
        or (flags["missing_second_attacker"] and int(root["turn"]) >= 4)
    )
    score = (
        10.0 * float(flags["prize_threshold"])
        + 8.0 * float(flags["attack_setup_tradeoff"])
        + 6.0 * float(flags["mewtwo_activation"])
        + 4.0 * float(flags["missing_second_attacker"])
        + 3.0 * float(flags["spidops_engine"])
        + 2.0 * float(flags["low_policy_margin"])
        + 0.10 * float(root["turn"])
        - min(3.0, margin if math.isfinite(margin) else 3.0)
    )
    return {
        "eligible": eligible,
        "score": score,
        "flags": flags,
        "ready_attacker_count": ready_count,
        "mewtwo_on_board": mewtwo_on_board,
        "mewtwo_ready": mewtwo_ready,
        "spidops_count": spidops_count,
        "own_prizes_remaining": own_prizes,
        "opponent_prizes_remaining": opponent_prizes,
        "candidate_option_types": sorted(option_types),
    }


def choose_root(
    runtime: CriticalRootRuntime,
    roots: Sequence[Mapping[str, Any]],
    *,
    minimum_alternative_support: float,
) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, Any]] | None:
    ranked = []
    for root in roots:
        diagnostics = policy_diagnostics(
            runtime,
            root,
            minimum_alternative_support=minimum_alternative_support,
        )
        if len(diagnostics["candidate_indices"]) < 2:
            continue
        crit = criticality(root, diagnostics)
        if not crit["eligible"]:
            continue
        # Highest criticality wins.  Exact ties prefer the later decision in
        # the source match, then the lower observation fingerprint.
        rank = (
            float(crit["score"]),
            int(root["turn"]),
            int(root["step"]),
            str(root["observation_fingerprint"]),
        )
        ranked.append((rank, root, diagnostics, crit))
    if not ranked:
        return None
    _, root, diagnostics, crit = max(ranked, key=lambda item: item[0])
    return root, diagnostics, crit


def _validate_frozen_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    runtime = args.runtime.resolve()
    own = args.own_candidate.resolve()
    opponent = args.opponent.resolve()
    for path in (runtime, own, opponent):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if tree_sha256(own) != OWN_TREE_SHA256:
        raise ValueError("frozen Wobbuffet-free candidate tree changed")
    if sha256(own / "deck.csv") != OWN_DECK_SHA256:
        raise ValueError("frozen Wobbuffet-free deck changed")
    if sha256(own / "model.pt") != OWN_MODEL_SHA256:
        raise ValueError("frozen Wobbuffet-free model changed")
    if sha256(own / "main.py") != OWN_MAIN_SHA256:
        raise ValueError("frozen Wobbuffet-free main changed")
    own_deck = [int(line) for line in (own / "deck.csv").read_text().splitlines() if line.strip()]
    if len(own_deck) != 60 or WOBBUFFET in own_deck:
        raise ValueError("pilot requires an exact 60-card deck with zero Wobbuffet")
    if tree_sha256(opponent) != MARNIE_TREE_SHA256:
        raise ValueError("frozen Marnie opponent tree changed")
    if sha256(opponent / "deck.csv") != MARNIE_DECK_SHA256:
        raise ValueError("frozen Marnie deck changed")
    if sha256(opponent / "main.py") != MARNIE_MAIN_SHA256:
        raise ValueError("frozen Marnie main changed")
    return runtime, own, opponent


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    runtime_path, own, opponent = _validate_frozen_inputs(args)
    runtime = CriticalRootRuntime(runtime_path, own, opponent, args.candidates)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    started = time()
    retained = 0
    attempts = 0
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
            selected = choose_root(
                runtime,
                capture["roots"],
                minimum_alternative_support=args.minimum_alternative_support,
            )
            if selected is None:
                counters["source_games_without_eligible_root"] += 1
                continue
            root, diagnostics, crit = selected
            root = dict(root)
            candidate_indices = [int(index) for index in diagnostics["candidate_indices"]]
            root["candidate_indices"] = candidate_indices
            options = (root["observation"].get("select") or {}).get("option") or ()
            root["candidate_options"] = {
                str(index): option_summary(options[index], root["observation"])
                for index in candidate_indices
            }
            evaluation = runtime.evaluate_belief_root(
                root,
                particles=args.particles,
                particle_offset=args.particle_offset,
                coin_rollouts=args.coin_rollouts,
                max_rollout_decisions=args.max_rollout_decisions,
                budget_seconds=args.root_budget_seconds,
                seed=args.seed + 100_003 * (retained + 1),
            )
            retained += 1
            baseline_utility = utility(int(capture["result"]), int(capture["own_seat"]))
            counters["retained_roots"] += 1
            counters["baseline_source_wins"] += int(baseline_utility > 0)
            counters["baseline_source_losses"] += int(baseline_utility < 0)
            counters["baseline_source_draws"] += int(baseline_utility == 0)
            counters["fully_complete_roots"] += int(evaluation["fully_complete"])
            counters["complete_pairs"] += int(evaluation["complete_pairs"])
            counters["attempted_pairs"] += int(evaluation["attempted_pairs"])
            counters["rollout_errors"] += len(evaluation["errors"])
            counters["guarded_roots"] += int(diagnostics["guard_name"] is not None)
            for name, active in crit["flags"].items():
                counters[f"critical_{name}"] += int(active)
            row = {
                "schema": "target80-paired-critical-root/v1",
                "protocol": "TARGET80_PAIRED_CRITICAL_ROOT_V1_PROTOCOL.md",
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
                "candidate_indices": candidate_indices,
                "candidate_options": root["candidate_options"],
                "observation_fingerprint": root["observation_fingerprint"],
                "observation": root["observation"],
                "policy_feature": diagnostics["policy_feature"],
                "policy_logits": diagnostics["policy_logits"],
                "policy_probabilities": diagnostics["policy_probabilities"],
                "policy_argmax": diagnostics["policy_argmax"],
                "guard_name": diagnostics["guard_name"],
                "guarded_baseline_reproduced": diagnostics["guarded_baseline_reproduced"],
                "policy_margin": diagnostics["policy_margin"],
                "criticality": crit,
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
                        "critical_flags": [name for name, value in crit["flags"].items() if value],
                        "complete_pairs": evaluation["complete_pairs"],
                        "fully_complete": evaluation["fully_complete"],
                        "errors": len(evaluation["errors"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if retained != args.roots:
        raise RuntimeError(
            f"retained only {retained}/{args.roots} roots in {attempts} source attempts"
        )
    manifest = {
        "schema": "target80-paired-critical-root-manifest/v1",
        "protocol": "TARGET80_PAIRED_CRITICAL_ROOT_V1_PROTOCOL.md",
        "method": "critical-state paired sampled-public-belief terminal continuations",
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
        "output": str(output),
        "output_sha256": sha256(output),
        "attempted_matches": attempts,
        "counts": dict(counters),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "elapsed_seconds": time() - started,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
