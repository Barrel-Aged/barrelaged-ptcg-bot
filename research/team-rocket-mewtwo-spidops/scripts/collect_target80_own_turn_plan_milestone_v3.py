#!/usr/bin/env python3
"""Compare complete Rocket plan lines at the end of the current own turn."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter, time
import random
import sys
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import collect_target80_paired_critical_roots_v1 as v1  # noqa: E402
import collect_target80_paired_milestone_roots_v2 as v2  # noqa: E402
from collect_live_truth_causal_action_values import (  # noqa: E402
    _valid_choice,
    option_summary,
    sha256,
    utility,
)
from collect_rocket_blueprint_n5_panel import tree_sha256  # noqa: E402


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
    parser.add_argument("--max-source-attempts", type=int, default=240)
    parser.add_argument("--max-live-decisions", type=int, default=2_000)
    parser.add_argument("--max-rollout-decisions", type=int, default=150)
    parser.add_argument("--root-budget-seconds", type=float, default=180.0)
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
    if not 0.0 < args.minimum_plan_action_support < 1.0:
        raise ValueError("--minimum-plan-action-support must be in (0, 1)")
    if args.particle_offset < 0 or args.root_budget_seconds <= 0.0:
        raise ValueError("invalid particle offset or root budget")
    if args.earliest_turn < 0 or args.latest_turn < args.earliest_turn:
        raise ValueError("invalid turn window")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite V3 artifact: {args.output}")


class OwnTurnMilestoneRuntime(v2.MilestoneRuntime):
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
        if horizon_turns != 1:
            raise ValueError("V3 requires the end-of-own-turn horizon marker 1")
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
                axes = v2.strategic_axes(raw, root_player)
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
            if turn >= root_turn + 1 and actor != root_player:
                axes = v2.strategic_axes(raw, root_player)
                return {
                    "complete": True,
                    "terminal": False,
                    "terminal_utility": None,
                    "value": v2.aggregate_axes(axes, weights),
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
                raise RuntimeError(f"V3 plan returned illegal action: {action}")
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


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    runtime_path, own, opponent = v1._validate_frozen_inputs(args)
    runtime = OwnTurnMilestoneRuntime(runtime_path, own, opponent, args.candidates)

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
            selected = v2.choose_milestone_root(
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
            plan_root_options = {
                name: option_summary(options[action[0]], root["observation"])
                for name, action in plan_actions.items()
            }
            evaluation = runtime.evaluate_milestone_root(
                root,
                critical=critical,
                particles=args.particles,
                particle_offset=args.particle_offset,
                coin_rollouts=args.coin_rollouts,
                horizon_turns=1,
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
                "schema": "target80-own-turn-plan-milestone/v3",
                "protocol": "TARGET80_OWN_TURN_PLAN_MILESTONE_V3_PROTOCOL.md",
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
                "plan_names": list(v2.PLAN_NAMES),
                "plan_weights": v2.PLAN_WEIGHTS,
                "plan_actions": plan_actions,
                "plan_root_options": plan_root_options,
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
        "schema": "target80-own-turn-plan-milestone-manifest/v3",
        "protocol": "TARGET80_OWN_TURN_PLAN_MILESTONE_V3_PROTOCOL.md",
        "method": "paired public-belief persistent plans to end of controllable own-turn line",
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
        "horizon": "first opponent observation after current Rocket turn completes",
        "opponent_actions_inside_horizon": False,
        "strategic_axes": list(v2.AXES),
        "strategic_plans": v2.PLAN_WEIGHTS,
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
        "v2_dependency_sha256": sha256(Path(v2.__file__).resolve()),
        "output": str(output),
        "output_sha256": sha256(output),
        "attempted_matches": attempts,
        "counts": dict(counters),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "elapsed_seconds": time() - started,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
