#!/usr/bin/env python3
"""Collect fresh paired roots for the frozen compact engine-resilience rule."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from time import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import collect_target80_paired_critical_roots_v1 as v1  # noqa: E402
import collect_target80_paired_milestone_roots_v2 as v2  # noqa: E402
import collect_target80_own_turn_plan_milestone_v3 as v3  # noqa: E402
from collect_live_truth_causal_action_values import option_summary, sha256, utility  # noqa: E402
from collect_rocket_blueprint_n5_panel import tree_sha256  # noqa: E402


PROTOCOL = "TARGET80_COMPACT_ENGINE_RULE_V5_TEST_PROTOCOL.md"


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
    parser.add_argument("--shard", choices=("test_first", "test_second"), required=True)
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
    parser.add_argument("--max-source-attempts", type=int, default=1_200)
    parser.add_argument("--max-live-decisions", type=int, default=2_000)
    parser.add_argument("--max-rollout-decisions", type=int, default=150)
    parser.add_argument("--root-budget-seconds", type=float, default=180.0)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v3.validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    runtime_path, own, opponent = v1._validate_frozen_inputs(args)
    runtime = v3.OwnTurnMilestoneRuntime(runtime_path, own, opponent, args.candidates)

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
            counters["retained_roots"] += 1
            counters["fully_complete_roots"] += int(evaluation["fully_complete"])
            counters["complete_pairs"] += int(evaluation["complete_pairs"])
            counters["attempted_pairs"] += int(evaluation["attempted_pairs"])
            counters["rollout_errors"] += len(evaluation["errors"])
            for name, active in critical["flags"].items():
                counters[f"critical_{name}"] += int(active)
            row = {
                "schema": "target80-compact-engine-rule-test-root/v5",
                "protocol": PROTOCOL,
                "shard": args.shard,
                "root_id": retained,
                "source_attempt": attempts,
                "opponent": opponent.name,
                "own_seat": capture["own_seat"],
                "started_first": capture["started_first"],
                "first_player": capture["first_player"],
                "baseline_source_result": capture["result"],
                "baseline_source_utility": utility(
                    int(capture["result"]), int(capture["own_seat"])
                ),
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
                "policy_margin": diagnostics["policy_margin"],
                "guard_name": diagnostics["guard_name"],
                "guarded_baseline_reproduced": diagnostics[
                    "guarded_baseline_reproduced"
                ],
                "criticality": critical,
                **evaluation,
            }
            sink.write(json.dumps(row, separators=(",", ":")) + "\n")
            sink.flush()
            print(
                json.dumps(
                    {
                        "shard": args.shard,
                        "root": retained,
                        "of": args.roots,
                        "source_attempt": attempts,
                        "complete_pairs": evaluation["complete_pairs"],
                        "fully_complete": evaluation["fully_complete"],
                        "errors": len(evaluation["errors"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if retained != args.roots:
        raise RuntimeError(f"retained only {retained}/{args.roots} roots")
    manifest = {
        "schema": "target80-compact-engine-rule-test-manifest/v5",
        "protocol": PROTOCOL,
        "shard": args.shard,
        "method": "fresh prospective paired test of frozen compact rule",
        "genuine_official_engine_source_matches": True,
        "genuine_official_engine_descendant_transitions": True,
        "source_policy_exact_packaged_public_guards": True,
        "visualizer_or_realized_hidden_truth_read": False,
        "hidden_completion_distribution_official": False,
        "raw_hidden_allocations_serialized": False,
        "common_random_coin_sequences": True,
        "sampled_rollout_is_final_strategy": False,
        "deployment_authority": False,
        "horizon": "first opponent observation after current Rocket turn completes",
        "opponent_actions_inside_horizon": False,
        "strategic_axes": list(v2.AXES),
        "strategic_plans": v2.PLAN_WEIGHTS,
        "runtime": str(runtime_path),
        "own_candidate": str(own),
        "opponent": str(opponent),
        "own_tree_sha256": tree_sha256(own),
        "own_deck_sha256": sha256(own / "deck.csv"),
        "own_model_sha256": sha256(own / "model.pt"),
        "own_main_sha256": sha256(own / "main.py"),
        "opponent_tree_sha256": tree_sha256(opponent),
        "collector_sha256": sha256(Path(__file__).resolve()),
        "v3_dependency_sha256": sha256(Path(v3.__file__).resolve()),
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
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
