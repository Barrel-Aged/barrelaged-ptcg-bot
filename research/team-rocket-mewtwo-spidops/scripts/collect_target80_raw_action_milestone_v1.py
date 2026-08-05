#!/usr/bin/env python3
"""Collect paired end-of-own-turn values for raw legal actions.

Source matches and every descendant transition use the official ``cg`` engine.
Each candidate action is evaluated under the same public-consistent hidden-zone
particle and the same manual coin sequence.  Continuations use the frozen,
guarded blueprint only until the current Rocket turn ends, then the six public
strategic axes are combined by the state-conditioned meta value.

This is offline teacher-data collection.  It is not a deployed rollout policy,
an equilibrium solver, an official posterior, or a safety certificate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter, time
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import collect_target80_paired_critical_roots_v1 as v1  # noqa: E402
import collect_target80_paired_milestone_roots_v2 as v2  # noqa: E402
from collect_belief_rollout_teacher_n6 import BeliefRolloutRuntime, _mean_and_se  # noqa: E402
from collect_live_truth_causal_action_values import (  # noqa: E402
    DEFAULT_EVALUATOR,
    _valid_choice,
    option_summary,
    sha256,
    utility,
)
from collect_rocket_blueprint_n5_panel import tree_sha256  # noqa: E402


EXPECTED_DECK_SHA256 = "0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8"
EXPECTED_MODEL_SHA256 = "b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f"
EXPECTED_MAIN_SHA256 = "3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341"
WOBBUFFET = 432


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
        / "hancang_wobbfree_elite_bc_v1r_selected",
    )
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--opponent-timeout", type=float, default=30.0)
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
    parser.add_argument("--latest-turn", type=int, default=8)
    parser.add_argument("--max-source-attempts", type=int, default=240)
    parser.add_argument("--max-live-decisions", type=int, default=2_000)
    parser.add_argument("--max-rollout-decisions", type=int, default=150)
    parser.add_argument("--root-budget-seconds", type=float, default=180.0)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
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
    if args.opponent_timeout <= 0.0:
        raise ValueError("--opponent-timeout must be positive")
    if args.earliest_turn < 0 or args.latest_turn < args.earliest_turn:
        raise ValueError("invalid turn window")

    runtime = args.runtime.resolve()
    own = args.own_candidate.resolve()
    opponent = args.opponent.resolve()
    evaluator = args.evaluator.resolve()
    for path in (runtime, own, opponent):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)
    expected = {
        own / "deck.csv": EXPECTED_DECK_SHA256,
        own / "model.pt": EXPECTED_MODEL_SHA256,
        own / "main.py": EXPECTED_MAIN_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"frozen own identity mismatch: {path}")
    own_deck = [int(value) for value in (own / "deck.csv").read_text().replace(",", "\n").split()]
    if len(own_deck) != 60 or WOBBUFFET in own_deck:
        raise ValueError("collector requires an exact 60-card, zero-Wobbuffet deck")
    for name in ("deck.csv", "main.py"):
        if not (opponent / name).is_file():
            raise FileNotFoundError(opponent / name)
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite raw-action artifact: {output}")
    return runtime, own, opponent, evaluator


def _best_action(
    paired: Sequence[Mapping[str, Any]],
    candidates: Sequence[int],
    baseline: int,
    *,
    particles: set[int] | None = None,
    coin: int | None = None,
) -> int | None:
    selected = [
        row
        for row in paired
        if (particles is None or int(row["particle"]) in particles)
        and (coin is None or int(row["coin_rollout"]) == coin)
    ]
    if not selected:
        return None
    means = {
        index: statistics.fmean(float(row["values"][str(index)]) for row in selected)
        for index in candidates
    }
    order = {index: position for position, index in enumerate(candidates)}
    return max(
        candidates,
        key=lambda index: (
            means[index],
            int(index == baseline),
            -order[index],
        ),
    )


class RawActionMilestoneRuntime(BeliefRolloutRuntime):
    """Official runtime exposing exact packaged-guard diagnostics and dense leaves."""

    def __init__(
        self,
        runtime: Path,
        own: Path,
        opponent: Path,
        candidates: int,
        *,
        evaluator: Path,
        opponent_timeout: float,
    ) -> None:
        super().__init__(
            runtime,
            own,
            opponent,
            candidates,
            evaluator=evaluator,
            opponent_timeout=opponent_timeout,
        )

    def guard_action(self, observation: Mapping[str, Any]) -> tuple[list[int] | None, str | None]:
        public_observation = dict(observation)
        for name, guard in self.guards:
            action = guard(public_observation)
            if action is not None:
                return [int(index) for index in action], str(name)
        return None, None

    def _horizon_rollout(
        self,
        branch: Any,
        *,
        root_player: int,
        root_turn: int,
        beliefs: tuple[Any, Any],
        opponent_history: Sequence[tuple[Mapping[str, Any], Sequence[int]]],
        weights: Mapping[str, float],
        max_decisions: int,
        deadline: float,
        coin_seed: int,
    ) -> dict[str, Any]:
        agents = self.new_agents(beliefs, opponent_history=opponent_history)
        actors = agents if root_player == 0 else (agents[1], agents[0])
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
                raise RuntimeError(f"raw-action horizon policy returned illegal action: {action}")
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

    def evaluate_action_root(
        self,
        root: Mapping[str, Any],
        *,
        critical: Mapping[str, Any],
        particles: int,
        particle_offset: int,
        coin_rollouts: int,
        max_rollout_decisions: int,
        budget_seconds: float,
        seed: int,
    ) -> dict[str, Any]:
        deadline = perf_counter() + budget_seconds
        candidates = tuple(int(value) for value in root["candidate_indices"])
        baseline = int(root["baseline_action"][0])
        weights = v2.meta_weights(root, critical)
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
                    order = list(candidates)
                    random.Random(seed + 10_007 * particle + coin_rollout).shuffle(order)
                    coin_seed = seed + 65_537 * particle + 977 * coin_rollout
                    for index in order:
                        if perf_counter() >= deadline:
                            break
                        try:
                            child = self.search_step(search_root.searchId, [index])
                            outcome = self._horizon_rollout(
                                child,
                                root_player=int(root["_own_seat"]),
                                root_turn=int(root["turn"]),
                                beliefs=root["_beliefs"],
                                opponent_history=root["_opponent_history"],
                                weights=weights,
                                max_decisions=max_rollout_decisions,
                                deadline=deadline,
                                coin_seed=coin_seed,
                            )
                            if outcome["complete"]:
                                key = str(index)
                                values[key] = float(outcome["value"])
                                axes[key] = dict(outcome["axes"])
                                terminals[key] = bool(outcome["terminal"])
                                terminal_utilities[key] = outcome["terminal_utility"]
                                depths[key] = int(outcome["depth"])
                                horizon_reached[key] = int(outcome["horizon_turn"])
                                coin_flips[key] = int(outcome["coin_flips"])
                        except Exception as error:
                            errors.append(
                                f"particle={particle} coin={coin_rollout} candidate={index}: "
                                f"{type(error).__name__}: {str(error)[:180]}"
                            )
                    if len(values) == len(candidates):
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
            index: [float(row["values"][str(index)]) for row in paired]
            for index in candidates
        }
        means = {index: _mean_and_se(values)[0] for index, values in samples.items()}
        standard_errors = {index: _mean_and_se(values)[1] for index, values in samples.items()}
        best = _best_action(paired, candidates, baseline) or baseline
        deltas = {
            index: [
                float(row["values"][str(index)]) - float(row["values"][str(baseline)])
                for row in paired
            ]
            for index in candidates
        }
        particle_ids = list(range(particle_offset, particle_offset + particles))
        midpoint = len(particle_ids) // 2
        particle_half_best = [
            _best_action(paired, candidates, baseline, particles=set(particle_ids[:midpoint])),
            _best_action(paired, candidates, baseline, particles=set(particle_ids[midpoint:])),
        ]
        coin_best = [
            _best_action(paired, candidates, baseline, coin=coin)
            for coin in range(coin_rollouts)
        ]
        stable_by_particles = all(value == best for value in particle_half_best)
        stable_by_coins = all(value == best for value in coin_best)
        best_deltas = deltas.get(best, [])
        return {
            "baseline": baseline,
            "best": best,
            "changed": best != baseline,
            "means": {str(index): value for index, value in means.items()},
            "standard_errors": {str(index): value for index, value in standard_errors.items()},
            "paired_deltas": {str(index): values for index, values in deltas.items()},
            "best_mean_paired_delta": _mean_and_se(best_deltas)[0] if best_deltas else 0.0,
            "best_delta_standard_error": _mean_and_se(best_deltas)[1] if best_deltas else None,
            "best_negative_fraction": (
                sum(value < -1e-9 for value in best_deltas) / len(best_deltas)
                if best_deltas
                else 0.0
            ),
            "particle_half_best": particle_half_best,
            "coin_best": coin_best,
            "stable_by_particles": stable_by_particles,
            "stable_by_coins": stable_by_coins,
            "stable_label": stable_by_particles and stable_by_coins,
            "paired_samples": paired,
            "complete_pairs": len(paired),
            "attempted_pairs": attempted_pairs,
            "fully_complete": len(paired) == particles * coin_rollouts,
            "particle_manifests": particle_manifests,
            "meta_weights": weights,
            "errors": errors,
        }


def main() -> int:
    args = parse_args()
    runtime_path, own, opponent, evaluator = validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    runtime = RawActionMilestoneRuntime(
        runtime_path,
        own,
        opponent,
        args.candidates,
        evaluator=evaluator,
        opponent_timeout=args.opponent_timeout,
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    retained = attempts = 0
    started = time()
    try:
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
                selected = v1.choose_root(
                    runtime,
                    capture["roots"],
                    minimum_alternative_support=args.minimum_alternative_support,
                )
                if selected is None:
                    counters["source_games_without_eligible_root"] += 1
                    continue
                root, diagnostics, critical = selected
                root = dict(root)
                candidate_indices = [int(index) for index in diagnostics["candidate_indices"]]
                root["candidate_indices"] = candidate_indices
                options = (root["observation"].get("select") or {}).get("option") or ()
                candidate_options = {
                    str(index): option_summary(options[index], root["observation"])
                    for index in candidate_indices
                }
                evaluation = runtime.evaluate_action_root(
                    root,
                    critical=critical,
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
                counters["fully_complete_roots"] += int(evaluation["fully_complete"])
                counters["stable_roots"] += int(evaluation["stable_label"])
                counters["stable_changes"] += int(
                    evaluation["stable_label"] and evaluation["changed"]
                )
                counters["complete_pairs"] += int(evaluation["complete_pairs"])
                counters["attempted_pairs"] += int(evaluation["attempted_pairs"])
                counters["rollout_errors"] += len(evaluation["errors"])
                counters["guarded_roots"] += int(diagnostics["guard_name"] is not None)
                for name, active in critical["flags"].items():
                    counters[f"critical_{name}"] += int(active)
                row = {
                    "schema": "target80-raw-action-milestone/v1",
                    "protocol": "TARGET80_RAW_ACTION_MILESTONE_POLICY_V1_PROTOCOL.md",
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
                    "candidate_options": candidate_options,
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
                            "best": evaluation["best"],
                            "baseline": evaluation["baseline"],
                            "stable": evaluation["stable_label"],
                            "delta": evaluation["best_mean_paired_delta"],
                            "complete_pairs": evaluation["complete_pairs"],
                            "errors": len(evaluation["errors"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        runtime.close()

    if retained != args.roots:
        raise RuntimeError(f"retained only {retained}/{args.roots} roots in {attempts} attempts")
    manifest = {
        "schema": "target80-raw-action-milestone-manifest/v1",
        "protocol": "TARGET80_RAW_ACTION_MILESTONE_POLICY_V1_PROTOCOL.md",
        "method": "paired raw legal actions to end of controllable own-turn line",
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
        "leaf_value_substitution": True,
        "sampled_rollout_is_final_strategy": False,
        "deployment_authority": False,
        "horizon": "first opponent observation after current Rocket turn completes",
        "opponent_actions_inside_horizon": False,
        "strategic_axes": list(v2.AXES),
        "runtime": str(runtime_path),
        "own_candidate": str(own),
        "opponent": str(opponent),
        "own_tree_sha256": tree_sha256(own),
        "own_deck_sha256": sha256(own / "deck.csv"),
        "own_model_sha256": sha256(own / "model.pt"),
        "own_main_sha256": sha256(own / "main.py"),
        "opponent_tree_sha256": tree_sha256(opponent),
        "opponent_deck_sha256": sha256(opponent / "deck.csv"),
        "opponent_model_sha256": (
            sha256(opponent / "model.pt") if (opponent / "model.pt").is_file() else None
        ),
        "opponent_main_sha256": sha256(opponent / "main.py"),
        "collector_sha256": sha256(Path(__file__).resolve()),
        "v1_dependency_sha256": sha256(Path(v1.__file__).resolve()),
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
