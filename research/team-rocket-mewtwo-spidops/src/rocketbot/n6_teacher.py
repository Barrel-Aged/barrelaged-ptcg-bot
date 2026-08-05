"""Statistics for the N6 paired public-belief rollout teacher.

Utilities are paired by hidden-allocation particle and coin stream. Confidence
intervals resample particles, keeping both coin streams in the same cluster.
The module is independent of the licensed runtime and safe to unit test.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
import statistics
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ActionEstimate:
    action: int
    mean: float
    paired_difference: float
    clustered_standard_error: float | None
    clustered_interval_95: tuple[float, float] | None
    clusters: int


@dataclass(frozen=True, slots=True)
class RootTeacherEstimate:
    baseline: int
    best: int
    actions: tuple[ActionEstimate, ...]
    half_best: tuple[int | None, int | None]
    half_paired_differences: tuple[float | None, float | None]
    complete_pairs: int
    complete_clusters: int
    admissible_change: bool

    def action(self, action: int) -> ActionEstimate:
        for estimate in self.actions:
            if estimate.action == action:
                return estimate
        raise KeyError(action)


def _cluster_means(
    paired_samples: Sequence[Mapping[str, Any]],
    *,
    action: int,
    baseline: int,
) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in paired_samples:
        values = row.get("values") or {}
        action_key = str(action)
        baseline_key = str(baseline)
        if action_key not in values or baseline_key not in values:
            continue
        grouped[int(row["particle"])].append(
            float(values[action_key]) - float(values[baseline_key])
        )
    return [
        (particle, statistics.fmean(values))
        for particle, values in sorted(grouped.items())
        if values
    ]


def clustered_bootstrap_interval(
    values: Sequence[float],
    *,
    seed: int,
    replicates: int = 10_000,
) -> tuple[float, float] | None:
    """Percentile interval from an i.i.d. cluster bootstrap."""

    materialized = tuple(float(value) for value in values)
    if len(materialized) < 2:
        return None
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = random.Random(seed)
    count = len(materialized)
    means = []
    for _ in range(replicates):
        means.append(
            statistics.fmean(materialized[rng.randrange(count)] for _ in range(count))
        )
    means.sort()
    lower_index = max(0, min(len(means) - 1, int(math.floor(0.025 * len(means)))))
    upper_index = max(0, min(len(means) - 1, int(math.ceil(0.975 * len(means))) - 1))
    return means[lower_index], means[upper_index]


def _best_for_particles(
    paired_samples: Sequence[Mapping[str, Any]],
    candidates: Sequence[int],
    particles: set[int],
) -> int | None:
    selected = [row for row in paired_samples if int(row["particle"]) in particles]
    if not selected:
        return None
    means = {
        action: statistics.fmean(
            float(row["values"][str(action)])
            for row in selected
            if str(action) in (row.get("values") or {})
        )
        for action in candidates
    }
    return max(means, key=lambda action: (means[action], -action))


def estimate_root(
    row: Mapping[str, Any],
    *,
    seed: int,
    bootstrap_replicates: int = 10_000,
    minimum_complete_pairs: int = 22,
) -> RootTeacherEstimate:
    """Build the preregistered N6 estimate for one serialized root."""

    baseline = int(row["baseline"])
    candidates = tuple(int(value) for value in row["candidate_indices"])
    if baseline not in candidates:
        raise ValueError("baseline is absent from N6 candidate shortlist")
    paired_samples = tuple(row.get("paired_samples") or ())
    if not paired_samples:
        raise ValueError("N6 root has no complete paired samples")
    means = {
        action: statistics.fmean(
            float(sample["values"][str(action)]) for sample in paired_samples
        )
        for action in candidates
    }
    best = max(means, key=lambda action: (means[action], -action))
    estimates = []
    for position, action in enumerate(candidates):
        clustered = _cluster_means(
            paired_samples,
            action=action,
            baseline=baseline,
        )
        differences = [value for _, value in clustered]
        difference = statistics.fmean(differences) if differences else 0.0
        standard_error = (
            statistics.stdev(differences) / math.sqrt(len(differences))
            if len(differences) >= 2
            else None
        )
        interval = clustered_bootstrap_interval(
            differences,
            seed=seed + 10_007 * position,
            replicates=bootstrap_replicates,
        )
        estimates.append(
            ActionEstimate(
                action=action,
                mean=means[action],
                paired_difference=difference,
                clustered_standard_error=standard_error,
                clustered_interval_95=interval,
                clusters=len(clustered),
            )
        )
    particle_ids = sorted({int(sample["particle"]) for sample in paired_samples})
    midpoint = len(particle_ids) // 2
    half_best = (
        _best_for_particles(paired_samples, candidates, set(particle_ids[:midpoint])),
        _best_for_particles(paired_samples, candidates, set(particle_ids[midpoint:])),
    )
    best_clustered = dict(
        _cluster_means(paired_samples, action=best, baseline=baseline)
    )
    half_differences = tuple(
        statistics.fmean(best_clustered[particle] for particle in selected)
        if selected and all(particle in best_clustered for particle in selected)
        else None
        for selected in (particle_ids[:midpoint], particle_ids[midpoint:])
    )
    best_estimate = next(item for item in estimates if item.action == best)
    lower = (
        best_estimate.clustered_interval_95[0]
        if best_estimate.clustered_interval_95 is not None
        else -math.inf
    )
    admissible = bool(
        best != baseline
        and len(paired_samples) >= minimum_complete_pairs
        and all(value is not None and value >= 0.0 for value in half_differences)
        and lower > 0.0
    )
    return RootTeacherEstimate(
        baseline=baseline,
        best=best,
        actions=tuple(estimates),
        half_best=half_best,
        half_paired_differences=half_differences,
        complete_pairs=len(paired_samples),
        complete_clusters=len(particle_ids),
        admissible_change=admissible,
    )


def redistribution_coefficient(means: Sequence[float], complete_clusters: int) -> float:
    if not means or complete_clusters <= 0:
        return 0.0
    spread = max(means) - min(means)
    return max(0.0, min(1.0, spread)) * min(1.0, math.sqrt(complete_clusters / 12.0))
