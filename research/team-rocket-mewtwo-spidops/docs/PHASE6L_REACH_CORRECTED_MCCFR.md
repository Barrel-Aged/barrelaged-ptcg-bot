# Phase 6L reach-correct external-sampling MCCFR reference

Phase 6L establishes the mathematical training reference that Phase 6K did not
provide. Phase 6K evaluates every action at a small number of selected native
information sets with independent terminal rollouts. That is useful wiring, but
it is not a Monte Carlo CFR traversal and does not carry a convergence result.

The new `ExternalSamplingTraversal` implements external-sampling MCCFR:

- at a traverser information set, expand every legal action;
- at chance and opponent information sets, sample one action from the current
  strategy, reusing one opponent action per information set within the block;
- propagate terminal utility back through the sampled block;
- record `value(action) - strategy-weighted value` for every traverser action;
- collect the opposing player's current policy in a separate strategy-memory
  record when such a node is encountered; and
- discard every partial sample if a transition or depth limit is reached.

These are conditional external-sampling advantages. They deliberately do not
carry an additional counterfactual-reach multiplier: the probability that the
sampled block visits the information set supplies that reach factor in
expectation.

## Primary-source check

The implementation was checked against:

- [Lanctot et al., Monte Carlo Sampling for Regret Minimization in Extensive Games](https://proceedings.neurips.cc/paper_files/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf), especially the external-sampling construction and Equation 11;
- [the corrected MCCFR supplemental material](https://webdocs.cs.ualberta.ca/~bowling/papers/09nips-mccfr-appendix.pdf), including the full sampled-regret derivation and algorithm; and
- [Brown et al., Deep Counterfactual Regret Minimization](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf), especially Algorithms 1–2 and the separate advantage/average-strategy memories.

Phase 6L does not call its implementation Deep CFR. No neural advantage model
is trained in this milestone.

## Exact expectation audit

`RocketPrizeDuel` permits exhaustive enumeration of every root deal and every
opponent pure-strategy sample. For a uniform target profile, the audit covers
384 positive sampled blocks for each traverser, 768 total. All 24
information-set/action rows agree with their exactly computed conditional
advantages.

| Check | Result |
|---|---:|
| Information-set/action rows | 24 |
| Player-0 sampled blocks | 384 |
| Player-1 sampled blocks | 384 |
| Maximum absolute expectation error | `9.992007221626409e-16` |
| Audit fingerprint | `9a1b3aef042b3d609b8151e23a2bf2d2ea693569fe57c188f01082c2c4cc6b1b` |

The exact audit is the important result: it guards against replacing
counterfactual reach with local rollout deltas or applying an extra reach
factor to already conditional Deep-CFR-style samples.

## Tabular convergence sanity check

A seeded tabular solver then runs one external-sampling traversal per player on
each of 20,000 iterations. The average strategy is evaluated by exact best
response because the reference game is enumerable.

| Metric | Result |
|---|---:|
| Seed | 6,019 |
| Iterations / traversals | 20,000 / 40,000 |
| Transitions | 210,852 |
| Advantage samples | 48,933 |
| Strategy samples | 72,986 |
| Uniform NashConv | 0.916666667 |
| Trained NashConv | 0.004617497 |
| Trained exploitability | 0.002308749 |
| Player-0 value | -0.055483855 |
| Solver fingerprint | `378945361592666ca094ea7b6e2c32fd2e0c2a8a9f92cdf8b6f480ada76bcd9b` |

This is convergence evidence only for the abstract 55-state validation game.
It is not Pokémon TCG performance evidence.

## Reservoir and traversal contracts

`AdvantageReservoir` and `StrategyReservoir` use deterministic SHA-256
priorities to retain bounded uniform samples. Iteration remains explicit on
each record for future LCFR-weighted loss, but does not change reservoir
admission. Both memories serialize with fingerprints and reject priority or
sample tampering.

The generic game boundary now exposes `player_to_act(state)` so a traversal
does not inspect solver-private state fields. Optional information-view and
action adapters let the native bridge supply redacted semantic features while
mapping enriched legal actions back to exact native selections.

## Durable artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| `expectation_audit.json` | 8,204 | `91272c282e69db3ddc81986668f32d616fe412bd147b390ee514cec694ecfd30` |
| `training_result.json` | 2,604 | `1aced055048b3b975fd63cceb17e635d9c0a878be6b32f5bd88fce7c7958533f` |
| `phase6l_evidence.json` | 1,804 | `c7be5894ee4c4b572bd16dd6531575405630836a25d32e4be5d64e6fa735f3d2` |
| `phase6l_evidence.md` | 1,995 | `249a33a0dc0e9d3ef35aba06ae66c2e5ef64fe588f5d13b5529dc5f5806d1c05` |

The evidence fingerprint is
`1f1fd025939174b9ea9ad53b713791e7db63f5f5f2c6b8647632bc442324fa55`.

## Remaining boundary

Phase 6L does not establish:

- a full-game Deep CFR iteration;
- a neural advantage or average-policy model trained from these memories;
- a calibrated or history-conditioned hidden-state prior;
- native full-game NashConv, exploitability, or best response;
- calibrated counterfactual-value bounds;
- safe subgame resolving or bounded exploitation; or
- a deployable or Kaggle-submitted policy.

Phase 6M exercises the native adapters on complete terminal endgames. Scaling
beyond that requires either tractable subgame boundaries with valid leaf values
or a carefully derived outcome-sampling/variance-reduced estimator; silently
depth-truncating external sampling would invalidate this reference.
