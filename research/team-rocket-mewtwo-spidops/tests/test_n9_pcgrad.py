from __future__ import annotations

import math

import torch

from scripts.train_live_outcome_awr import WeightedExample
from scripts.train_n9_balanced_pcgrad_awr import (
    GROUPS,
    apply_balanced_weights,
    symmetric_pcgrad,
)


def test_pcgrad_keeps_aligned_average() -> None:
    first = (torch.tensor([1.0, 2.0]),)
    second = (torch.tensor([3.0, 4.0]),)
    combined, conflict, cosine = symmetric_pcgrad(first, second)
    assert not conflict
    assert cosine > 0
    assert torch.equal(combined[0], torch.tensor([2.0, 3.0]))


def test_pcgrad_projects_conflicting_gradients() -> None:
    first = (torch.tensor([1.0, 0.0]),)
    second = (torch.tensor([-1.0, 1.0]),)
    combined, conflict, cosine = symmetric_pcgrad(first, second)
    assert conflict
    assert cosine < 0
    # The resulting descent direction is non-adverse to either original task.
    assert float((combined[0] * first[0]).sum()) >= 0.0
    assert float((combined[0] * second[0]).sum()) >= 0.0


def test_outcome_weights_have_equal_group_mass() -> None:
    examples = []
    for group, outcomes in zip(GROUPS, ((True, False, False), (True, True, False))):
        for index, won in enumerate(outcomes):
            examples.append(WeightedExample({}, 0, f"{group}-{index}", won, group))
    masses = apply_balanced_weights(
        examples,
        {GROUPS[0]: 1.0 / 3.0, GROUPS[1]: 2.0 / 3.0, "__global__": 0.5},
        0.5,
    )
    assert math.isclose(masses[GROUPS[0]], masses[GROUPS[1]], rel_tol=0, abs_tol=1e-6)
    assert math.isclose(sum(masses.values()), len(examples), rel_tol=0, abs_tol=1e-6)
