# Phase 6AL preregistration: high-precision conditional-value confirmation

Status: **protocol frozen before any Phase 6AL official match or replay batch**

## Question

Phase 6AK validates a collection-heldout action-sensitive point model and
passes aggregate 90% coverage and mean-width gates, but its worst declared
slice covers only `0.666667` against the fixed `0.75` floor. Post-hoc radius
widening is not admissible: it was chosen after validation was opened and does
not transfer when the old calibration and validation panels are reversed.

Phase 6AL tests whether lower-noise action targets plus a conservatively higher,
label-free calibration quantile can satisfy the existing interval gates. It
does not change or retrain the Phase 6AK mean model, use true intent as input,
or weaken any admission threshold.

This remains a supplied-range, abstract five-plan conditional-value
experiment. It is not an official hidden-state prior, opponent
counterfactual-value calibration, equilibrium blueprint, safe resolver,
official controller, Kaggle submission, or playing-strength experiment.

## Frozen model and development ledger

The exact Phase 6AK model fingerprint is fixed:

`5f3f73025251fde4d5e1dc155f604aa98856951030567e8baa13d2f1e999350e`

Its 208 pre-action root features, residual ridge mean, and centered causal
public branch heuristic are unchanged. All 288 Phase 6AI, Phase 6AJ, and Phase
6AK roots are now declared development evidence. Phase 6AK calibration and
validation outcomes motivated this protocol and cannot support the Phase 6AL
claim.

No intent label, policy-implementation label, split label, target return,
target standard error, selected hidden truth, raw hidden-zone identity,
post-branch private state, or future event becomes model input.

## High-precision replay bundle

Each fresh retained official root is replayed in two independently seeded
batches. Each batch uses two independently sampled public-consistent hidden
allocations, all five plan actions, and eight paired terminal continuations per
allocation/action. The bundle therefore contains:

- four supplied hidden allocations;
- twenty immediate public branch observations;
- five action targets;
- thirty-two terminal returns per action; and
- 160 terminal continuations per retained root.

The root action target is the mean of all 32 returns. Its standard error is the
sample standard deviation divided by `sqrt(32)`. Branch features and the causal
branch heuristic are averaged across the two replay batches. Both batches must
replay the same retained public state, acting-player private observation,
history features, blueprint distribution, and legal action ordering. Completion
fingerprints must be distinct across all four allocations.

Complete matches, legal selections, retained snapshots, native replay,
immediate branches, and descendants are genuine official-engine execution.
The hidden-allocation law, five-plan boundary, tactical executor, continuation
policy, fixed heuristic, mean model, bootstrap, and intervals are abstractions.

## Fresh sequential panels

The official runtime is unseeded. Abstraction-side allocation and continuation
sampling uses frozen seeds and namespaces:

- calibration: seed `6056`, namespace `phase6al-calibration`;
- validation: seed `6057`, namespace `phase6al-validation`; and
- contrast bootstrap: seed `6058`, 10,000 root resamples.

Replay batch zero uses the panel seed. Replay batch one uses the deterministic
derived seed `panel_seed + 1,000,000`; the match ID is also included in every
allocation and continuation seed derivation.

Each panel contains one fresh `action_first` retained root for every cross of
the six candidate decks and six declared intent-policy families: 36 matches,
720 immediate branch observations, 180 action targets, and 5,760 terminal
continuations. Across both panels the protocol requires 72 matches, 1,440
branches, 360 targets, and 11,520 continuations.

The Phase 6AK model reference must be serialized and fingerprint-checked before
calibration collection. The calibration panel must then be complete and its
bounds serialized before any validation worker may start. The 72 new root IDs
must be mutually distinct and disjoint from all 288 declared development roots.

## Frozen conservative calibration

For every calibration action target, the nonconformity score is:

`abs(prediction - empirical_mean) + 1.645 * target_standard_error`.

The global radius is the finite-sample 92% split-conformal quantile: rank
`ceil((180 + 1) * 0.92) = 167` in sorted ascending scores, clipped only to the
utility range. It is public/model-output conditioned only through the frozen
prediction and does not use candidate, intent, or policy labels to choose a
radius. The 92% target is more conservative than Phase 6AK's 90%; validation
acceptance thresholds below do not change.

## Frozen admission gates

The validation panel must satisfy every point and precision gate:

1. point MSE is lower than zero, the pre-root heuristic, the standalone causal
   branch heuristic, and the frozen Phase 6AJ hybrid on the same new roots;
2. action-contrast MSE is at least 2% lower than zero contrast;
3. action-contrast MSE is no more than 1% worse than the standalone branch
   heuristic;
4. the seed-`6058` 10,000-resample root bootstrap has a strictly positive 90%
   lower bound for zero-minus-model contrast error;
5. maximum 32-return target standard error is at most `0.20`; and
6. model-development, calibration, and validation root ledgers are mutually
   disjoint, with calibration frozen before validation.

Conditional expected-mean admission additionally retains every Phase 6AK gate:

- overall mean-CI coverage at least `0.90`;
- mean clipped interval width at most `1.75`; and
- every action, candidate, and declared-intent slice coverage at least `0.75`.

No threshold or model component may change after validation is opened.
Selected-action value/regret and pairwise concordance remain diagnostics.

## Fail-closed boundary

Regardless of outcome, Phase 6AL keeps false:

- official hidden-allocation prior or full public belief;
- human latent-intent calibration;
- opponent counterfactual-value compatibility;
- equilibrium convergence;
- safe-resolver enablement;
- opponent-exploitation enablement;
- official strategy control;
- Kaggle submission; and
- superhuman or other playing-strength claims.

A pass would admit only this conditional expected-mean provider under the
declared supplied-allocation/five-plan distribution. A later, independently
preregistered phase must still combine an official public range with
opponent-reach semantics and calibrate per-hypothesis counterfactual bounds
before any safe resolver can run.

## Frozen result

Status: **rejected under the preregistered gates; downstream remains fail
closed**.

Collection followed the required order. The model reference was serialized
before any Phase 6AL root, all 36 calibration roots were collected before the
calibration panel and radius were serialized, and only then were the 36
validation roots collected. The panels contain 72 distinct official matches,
1,440 immediate native branches, 360 action targets, and 11,520 terminal native
continuations. Every target has exactly 32 returns; maximum target standard
error is `0.179605`, below the `0.20` precision ceiling. Calibration and
validation root IDs are mutually disjoint and disjoint from the 288-root
development ledger. All 36 calibration captures succeeded on their first
attempt; 33 validation captures succeeded on attempt one and three used a
later permitted attempt.

The calibration panel fixed radius `1.0152108165` before validation. On the
held-out validation panel, point MSE `0.430678` beats zero `0.664497`, the root
heuristic `0.661890`, the branch heuristic `0.630282`, and the frozen Phase 6AJ
hybrid `0.615516`. Action-contrast MSE `0.009160` improves on zero contrast
`0.009583` and exactly retains the standalone branch contrast. Those point
comparisons pass, but the root-bootstrap 90% lower bound for the zero-minus-
model contrast improvement is `-0.000512`, so action-sensitive point admission
fails its preregistered confidence gate.

Overall interval coverage is `0.888889`, below `0.90`; mean width `1.566988`
passes `1.75`; and minimum declared-slice coverage is `0.666667`, below `0.75`.
The two worst slices are candidate `rocket_mewtwo` and declared intent
`combo_assembly`, each at `0.666667`. Conditional expected-mean admission is
therefore rejected independently by both aggregate and slice coverage.

No radius, model, threshold, split, or bootstrap rule was changed after
validation opened. The saved evidence status is
`high_precision_confirmation_rejected_downstream_fail_closed`; counterfactual
compatibility, equilibrium convergence, safe resolving, opponent exploitation,
official strategy control, Kaggle submission, and strength claims all remain
false.

### Frozen fingerprints

- model reference: `93bc23ba367dc8b606deb2aa829b7fa5b38e530bb602e6b39e12a76bae98869e`;
- source mean model: `5f3f73025251fde4d5e1dc155f604aa98856951030567e8baa13d2f1e999350e`;
- calibration panel: `cafc95994dabd9ed771b0b9b7630920d574e075e9ebc470a79fc528af05ee8e8`;
- calibration: `738bce39ad20a2951964579ba4d280d67339d00257ac2624b2dd08c25988bd9e`;
- validation panel: `c930633ed705c7fa26f2797b658175d63e7a36450b23e5804e99b432111bd201`;
- evaluation: `1bbcec7b1db62e3fdc451807dd9c6c067b341aa7fa4d915e089d424a7b53fa7d`;
- evidence: `d896278588650ed424bf47870990a9ddcf7dd27594cbc1a1d759af5f84763663`.
