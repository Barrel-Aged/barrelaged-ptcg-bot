#!/usr/bin/env python3
"""Collect paired public-belief rollout targets for N6 policy distillation.

Source games and every descendant transition use the official ``cg`` engine.
Hidden-zone allocations are sampled from the acting player's normal
observation and frozen deck priors. The native visualizer and realized hidden
state are never read. Raw sampled allocations remain process-local.

This is an offline teacher data collector. It is not an information-set solver,
an equilibrium certificate, or deployment authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
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
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collect_live_truth_causal_action_values import (  # noqa: E402
    OfficialCausalRuntime,
    fingerprint,
    sha256,
    utility,
)
from rocketbot.official_search_game import sample_hidden_zone_completion  # noqa: E402


DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "scale_neural_resolver_safe_tie_v3"
        ),
    )
    parser.add_argument(
        "--own-candidate",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "scale_value_calibrated_v1"
        ),
    )
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", choices=("first", "second", "balanced", "any"), default="second")
    parser.add_argument("--roots-per-game", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--particles", type=int, default=12)
    parser.add_argument("--coin-rollouts", type=int, default=2)
    parser.add_argument("--particle-offset", type=int, default=0)
    parser.add_argument("--earliest-turn", type=int, default=1)
    parser.add_argument("--latest-turn", type=int, default=8)
    parser.add_argument("--max-live-decisions", type=int, default=2_000)
    parser.add_argument("--max-rollout-decisions", type=int, default=600)
    parser.add_argument("--root-budget-seconds", type=float, default=180.0)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "games",
        "roots_per_game",
        "candidates",
        "particles",
        "coin_rollouts",
        "max_live_decisions",
        "max_rollout_decisions",
        "threads",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.particle_offset < 0:
        raise ValueError("--particle-offset must be non-negative")
    if args.root_budget_seconds <= 0:
        raise ValueError("--root-budget-seconds must be positive")
    if args.earliest_turn < 0 or args.latest_turn < args.earliest_turn:
        raise ValueError("invalid turn window")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite N6 shard: {args.output}")


def _mean_and_se(values: Sequence[float]) -> tuple[float, float | None]:
    if not values:
        return 0.0, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def _half_best(
    paired: Sequence[Mapping[str, Any]],
    candidates: Sequence[int],
    particle_ids: Sequence[int],
) -> int | None:
    selected = [row for row in paired if int(row["particle"]) in particle_ids]
    if not selected:
        return None
    means = {
        index: statistics.fmean(float(row["values"][str(index)]) for row in selected)
        for index in candidates
    }
    return max(means, key=lambda index: (means[index], -index))


class BeliefRolloutRuntime(OfficialCausalRuntime):
    """Official source matches plus abstract public-consistent particles."""

    def _completion(
        self,
        root: Mapping[str, Any],
        *,
        particle: int,
        total_particles: int,
        allocation_seed: int,
    ) -> Any:
        return sample_hidden_zone_completion(
            root["observation"],
            hypothesis_id=f"n6-particle-{particle:04d}",
            probability=1.0 / total_particles,
            allocation_seed=allocation_seed,
            your_candidate_id="team-rocket-mewtwo-spidops",
            opponent_candidate_id=self.opponent_path.name,
            your_archetype_id="team-rocket-box",
            opponent_archetype_id="marnie-general",
            your_style_id="frozen-neural-blueprint",
            opponent_style_id="frozen-neural-blueprint",
            your_card_ids=self.own_deck,
            opponent_card_ids=self.opponent_deck,
            your_deck_fingerprint=sha256(self.own_path / "deck.csv"),
            opponent_deck_fingerprint=sha256(self.opponent_path / "deck.csv"),
            manual_coin=True,
            source=(
                "N6 sampled public-consistent allocation from frozen deck priors; "
                "not official posterior truth"
            ),
        )

    def evaluate_belief_root(
        self,
        root: Mapping[str, Any],
        *,
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
                    depths: dict[str, int] = {}
                    coin_flips: dict[str, int] = {}
                    order = list(candidates)
                    random.Random(seed + 10_007 * particle + coin_rollout).shuffle(order)
                    coin_seed = seed + 65_537 * particle + 977 * coin_rollout
                    for index in order:
                        if perf_counter() >= deadline:
                            break
                        try:
                            child = self.search_step(search_root.searchId, [index])
                            value, terminal, depth, flips = self._rollout(
                                child,
                                root_player=int(root["_own_seat"]),
                                beliefs=root["_beliefs"],
                                max_decisions=max_rollout_decisions,
                                deadline=deadline,
                                coin_seed=coin_seed,
                                common_random_coins=True,
                            )
                            if terminal:
                                values[str(index)] = float(value)
                                depths[str(index)] = int(depth)
                                coin_flips[str(index)] = int(flips)
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
                                "depths": depths,
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
        best = max(means, key=lambda index: (means[index], -index)) if paired else baseline
        deltas = {
            index: [
                float(row["values"][str(index)]) - float(row["values"][str(baseline)])
                for row in paired
            ]
            for index in candidates
        }
        particle_ids = list(range(particle_offset, particle_offset + particles))
        midpoint = len(particle_ids) // 2
        half_best = (
            _half_best(paired, candidates, particle_ids[:midpoint]),
            _half_best(paired, candidates, particle_ids[midpoint:]),
        )
        return {
            "baseline": baseline,
            "best": best,
            "means": {str(index): value for index, value in means.items()},
            "standard_errors": {
                str(index): value for index, value in standard_errors.items()
            },
            "paired_deltas": {
                str(index): values for index, values in deltas.items()
            },
            "half_best": list(half_best),
            "paired_samples": paired,
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
    own = args.own_candidate.resolve()
    if sha256(own / "deck.csv") != DECK_SHA256:
        raise ValueError("N6 frozen Team Rocket deck changed")
    if sha256(own / "model.pt") != MODEL_SHA256:
        raise ValueError("N6 frozen initial model changed")
    runtime = BeliefRolloutRuntime(
        args.runtime,
        args.own_candidate,
        args.opponent,
        args.candidates,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    accepted_games = 0
    attempts = 0
    started = time()
    with output.open("x", encoding="utf-8") as sink:
        while accepted_games < args.games:
            attempts += 1
            capture = runtime.capture_match(
                seat_mode=args.seat,
                accepted_game_index=accepted_games + 1,
                roots_per_game=args.roots_per_game,
                earliest_turn=args.earliest_turn,
                latest_turn=args.latest_turn,
                max_decisions=args.max_live_decisions,
                common_random_coins=True,
                capture_truth=False,
            )
            if capture.get("seat_skipped"):
                counters["seat_skips"] += 1
                continue
            accepted_games += 1
            baseline_utility = utility(int(capture["result"]), int(capture["own_seat"]))
            counters["games"] += 1
            counters["baseline_wins"] += int(baseline_utility > 0)
            counters["baseline_losses"] += int(baseline_utility < 0)
            counters["games_without_roots"] += int(not capture["roots"])
            for root_index, root in enumerate(capture["roots"]):
                root_seed = args.seed + 100_003 * accepted_games + 3_001 * root_index
                evaluation = runtime.evaluate_belief_root(
                    root,
                    particles=args.particles,
                    particle_offset=args.particle_offset,
                    coin_rollouts=args.coin_rollouts,
                    max_rollout_decisions=args.max_rollout_decisions,
                    budget_seconds=args.root_budget_seconds,
                    seed=root_seed,
                )
                counters["roots"] += 1
                counters["fully_complete_roots"] += int(evaluation["fully_complete"])
                counters["complete_pairs"] += int(evaluation["complete_pairs"])
                counters["attempted_pairs"] += int(evaluation["attempted_pairs"])
                counters["rollout_errors"] += len(evaluation["errors"])
                row = {
                    "schema": "rocket-n6-belief-rollout-root/v1",
                    "game_id": accepted_games,
                    "capture_attempt": attempts,
                    "root_index": root_index,
                    "opponent": args.opponent.resolve().name,
                    "own_seat": capture["own_seat"],
                    "first_player": capture["first_player"],
                    "started_first": capture["started_first"],
                    "baseline_game_result": capture["result"],
                    "baseline_game_utility": baseline_utility,
                    "baseline_game_decisions": capture["decisions"],
                    "step": root["step"],
                    "turn": root["turn"],
                    "turn_action_count": root["turn_action_count"],
                    "baseline_action": root["baseline_action"],
                    "candidate_indices": root["candidate_indices"],
                    "candidate_options": root["candidate_options"],
                    "observation_fingerprint": root["observation_fingerprint"],
                    "observation": root["observation"],
                    **evaluation,
                }
                sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                sink.flush()
            print(
                json.dumps(
                    {
                        "game": accepted_games,
                        "of": args.games,
                        "baseline_utility": baseline_utility,
                        "roots": len(capture["roots"]),
                        "total_roots": counters["roots"],
                        "complete_roots": counters["fully_complete_roots"],
                        "complete_pairs": counters["complete_pairs"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    manifest = {
        "schema": "rocket-n6-belief-rollout-manifest/v1",
        "protocol": "N6_BELIEF_ROLLOUT_DISTILLATION_PROTOCOL.md",
        "method": "paired sampled-public-belief terminal rollout teacher",
        "genuine_official_engine_source_matches": True,
        "genuine_official_engine_descendant_transitions": True,
        "visualizer_or_realized_hidden_truth_read": False,
        "hidden_completion_distribution_official": False,
        "raw_hidden_allocations_serialized": False,
        "actor_information": "normal agent observation only",
        "public_information_set_expected_value": False,
        "equilibrium_or_safety_certificate": False,
        "common_random_coin_sequences": True,
        "leaf_value_substitution": False,
        "deployment_authority": False,
        "own_candidate": str(own),
        "opponent": str(args.opponent.resolve()),
        "own_model_sha256": sha256(own / "model.pt"),
        "opponent_model_sha256": sha256(args.opponent.resolve() / "model.pt"),
        "own_deck_sha256": sha256(own / "deck.csv"),
        "opponent_deck_sha256": sha256(args.opponent.resolve() / "deck.csv"),
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
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"complete": True, "counts": manifest["counts"], "manifest": str(manifest_path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

