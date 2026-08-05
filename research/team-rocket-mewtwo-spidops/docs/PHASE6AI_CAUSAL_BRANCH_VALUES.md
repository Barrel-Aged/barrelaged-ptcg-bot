# Phase 6AI preregistration: causal branch-conditioned values

Status: **fresh panel complete; neural action model rejected; downstream gates remain closed**

## Checked result

The frozen panel completed all 180 fresh official matches, 1,800 immediate
native branch observations, 360 supplied-allocation surfaces, 900 root/action
targets, and 14,400 terminal native continuations. It contains 19,270 genuine
live selections and 614,665 genuine native transitions. All captures succeeded
on attempt one. Dataset, model, calibration, and evidence fingerprints are:

| Artifact | Fingerprint |
|---|---|
| Dataset | `8a56e5f44d9fe3c3c680c5a586e02292e01bbc65c14ed3aa653c355ab312f2a8` |
| Model | `4c195b8dad7876ce03d2f433ba7d2393675d7fdcde9aedc9ea78e6e79ea18b7a` |
| Calibration | `1295b226009eeb130c9390b73972ce83f48049e5823c6f03419c4789ac0bee29` |
| Evidence | `27b440477f099edf4b6a8e63be99785b707d23a5ae04aa30c4a2c7970d02961b` |

Sampling precision passes again: maximum target SE is `0.258199`. The selected
64-hidden-unit model uses weight decay `0.01`, contrast weight `1.0`, mean blend
`0.50`, and advantage scale `1.0`. On untouched `action_first` roots, neural
point MSE `0.563726` beats zero `0.700087`, the pre-root heuristic `0.742190`,
and the causal branch heuristic `0.685935`.

The model nevertheless fails the action-sensitive gate. Its root-centered
contrast MSE `0.038066` is worse than zero contrast `0.037569` and the branch
heuristic's `0.035565`; the minimum preregistered bootstrap lower bound is
`-0.005853`. Pairwise concordance is `0.4875`, and its empirically selected
action value `0.017361` is lower than the branch heuristic's `0.062500`.

Intervals also fail: coverage is `0.877778`, mean width `1.870425`, and minimum
slice coverage `0.600000`, with Starmie the worst candidate slice. The
conditional expected-mean model is therefore rejected and all downstream
capabilities remain false.

The preregistered causal branch heuristic is a useful diagnostic: it improves
validation contrast MSE by 5.33% relative to zero. A post-result root bootstrap
gives a 90% lower bound of `0.000459` for that baseline improvement. Because a
standalone branch-baseline bootstrap was not an admission gate above, this is
hypothesis-generating evidence only, not a retroactive pass or deployed model.
Phase 6AJ subsequently preregistered and confirmed a frozen semiparametric use
of that ordering on a new root panel; see
`docs/PHASE6AJ_SEMIPARAMETRIC_BRANCH_HYBRID.md`. That later result does not
change this Phase 6AI rejection.

## Question

Phase 6AH made the conditional target precise enough to evaluate, but its
concatenated root/action MLP did not transfer action differences. The model saw
the pre-action public state, acting-player private observation, causal prefix,
candidate hypothesis, and an action one-hot. It did not see what the forced
plan actually changed before the frozen continuation blueprint began.

Phase 6AI tests one narrow next hypothesis: does a grouped action-relative
model generalize when every plan is represented by the immediate, public-only
child state produced by its genuine native branch?

This is not an equilibrium, opponent-counterfactual-value, safe-resolving,
deployment, leaderboard, or playing-strength experiment.

## Frozen independent panel

Phase 6AI must not reuse a Phase 6AH root or target. It uses seed `6049`, the
`phase6ai` match namespace, and the same complete implementation-isolated grid:

| Opponent implementation | Role | Replicas per candidate/intent |
|---|---|---:|
| `canonical` | train | 1 |
| `engine_first` | train | 1 |
| `hand_first` | calibration | 2 |
| `action_first` | untouched validation | 1 |

Every implementation is crossed with the six frozen candidate lists and six
declared intent families. This yields 180 complete matches and retained roots:
72 train, 36 calibration-selection, 36 calibration-bounds, and 36 untouched
validation. Root selection remains the third-to-last eligible player-0 `MAIN`
prompt after at least two opponent `MAIN` turns, with deterministic alternating
retry direction and no outcome filter.

Each retained root uses the Phase 6AH target protocol unchanged:

1. two independent public-consistent allocations from the realized candidate;
2. all five abstract turn-plan actions;
3. eight paired terminal continuations per allocation/action; and
4. sixteen terminal utilities per aggregated root/action target.

The frozen totals are 360 allocation surfaces, 900 root/action targets, and
14,400 terminal native continuations. Maximum target standard error must remain
at most `0.30`; rollout count is not increased to compensate for a weak model.

## Causal branch representation

For each allocation and forced plan, Phase 6AI records the immediate child
before any continuation rollout. Model-visible branch fields are restricted to:

- player-0-oriented public deltas for turn, action count, board count/HP,
  hand/deck/Prize counts, and publicly revealed opponent copies;
- L1 change summaries for own, opponent, and Stadium public card buckets;
- forced tactical transition count;
- whether the immediate child is terminal;
- the next boundary actor class (`self`, `opponent`, `chance`, or `terminal`);
- disagreement ranges across the two supplied allocations;
- the frozen blueprint probability for the plan; and
- the five-way plan identity.

The acting-player root-private features remain available. No post-branch hand,
deck, Prize, legal-option payload, selected hidden completion, intent label,
policy-implementation label, terminal continuation result, rollout depth, or
future continuation event may enter a model feature. Child state and public
payloads are persisted only as fingerprints and numerical allow-listed public
summaries. Both per-allocation descriptors remain auditable; the model consumes
their mean/range aggregation, never one selected hidden truth.

A fixed causal branch heuristic applies the existing Prize/HP/board formula to
the immediate public child. It is a required baseline, not a learned feature.

## Grouped dueling model

One training example is a complete root with all five actions. A shared root
encoder receives pre-action public, acting-private, causal-history, and
candidate-hypothesis features. A shared branch encoder receives one plan's
public child descriptor and plan identity. The network emits:

- one root-mean value; and
- five action advantages, centered to zero within the root.

Training minimizes root-mean squared error plus a frozen multiple of centered
action-contrast squared error. The output is the clipped sum of the root mean
and a shrinkable centered advantage. This structure cannot improve contrast by
moving every action together.

Only the training roots fit weights and normalizers. The fixed selection grid is:

- hidden width: `32`, `64`;
- weight decay: `1e-3`, `1e-2`, `1e-1`;
- contrast-loss weight: `1`, `4`;
- root-mean neural blend: `0.50`, `0.75`, `1.0`; and
- advantage scale: `0.10`, `0.25`, `0.50`, `1.0`.

Training uses 100 deterministic full-batch epochs at learning rate `0.01`.
`hand_first` replica 0 selects one configuration by the worst normalized point
and contrast error. Replica 1 is used only for interval construction.
`action_first` is read exactly once after the model and intervals are frozen.

## Admission gates

The fresh validation panel must satisfy every gate below:

1. point MSE is lower than zero, the pre-root heuristic, and the causal branch
   heuristic;
2. root-centered action-contrast MSE is at least 2% lower than both zero
   contrast and branch-heuristic contrast;
3. a deterministic 10,000-resample root bootstrap has a strictly positive 90%
   lower bound for baseline-minus-model contrast-MSE improvement;
4. maximum target standard error is at most `0.30`;
5. empirical-mean-CI coverage is at least `0.90`;
6. mean interval width is at most `1.75`; and
7. every action, candidate, and declared-intent slice has coverage at least
   `0.75`.

The bounds split constructs a finite-sample conformal interval around the
conditional mean prediction using
`abs(prediction - empirical_mean) + 1.645 * target_SE`, as in Phase 6AH.
Selected-action value/regret, pairwise concordance, and per-slice point errors
are diagnostics, not substitutes for the frozen gates.

## Provenance and fail-closed boundary

Genuine official-engine execution is limited to complete matches, legal live
selections, engine-issued retained snapshots, exact snapshot replay, forced
native descendants, and terminal native continuations.

The hidden-allocation law, plan actions, tactical executor, frozen continuation
blueprint, branch aggregation, heuristic, neural model, conformal intervals,
and bootstrap are research abstractions. Immediate child descriptors are based
on genuine native transitions, but the choice to expose those summaries to an
abstract five-plan model is not an official engine strategy.

Regardless of result, Phase 6AI keeps false:

- official hidden-allocation prior;
- opponent counterfactual-value compatibility;
- equilibrium convergence;
- safe-resolver enablement;
- opponent-exploitation enablement;
- official strategy control;
- Kaggle submission; and
- superhuman or other playing-strength claims.

Interrupted collection writes one fingerprinted root shard and resumes only
from schema-valid, grid-valid shards. Any provenance, leakage, split, shape,
fingerprint, or gate inconsistency is a hard failure.
