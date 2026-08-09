# Phase 6AK preregistration: expanded semiparametric root mean

Status: **protocol executed; action-sensitive point signal validated, conditional intervals rejected**

## Question

Phase 6AJ confirmed that a centered causal public branch heuristic carries a
transferable action-ordering signal, while its inherited 72-root neural state
mean left conditional intervals too wide. Phase 6AK tests whether a strongly
regularized state-mean model fitted on the complete prior development corpus
can sharpen those intervals without changing the confirmed action advantage.

This is a collection-heldout conditional expected-mean experiment. It is not
an implementation-heldout repeat, an official hidden-state prior, an opponent
counterfactual-value calibration, an equilibrium blueprint, safe resolving,
deployment, or a playing-strength claim.

## Frozen development corpus

The Phase 6AI dataset (180 roots) and Phase 6AJ fresh panel (36 roots) become a
declared 216-root development corpus. Their targets are no longer treated as
untouched evidence for Phase 6AK. The final claim comes only from new Phase 6AK
calibration and validation collections.

The model consumes the existing 208 named pre-action root features:

- public state;
- acting-player private observation;
- causal public history; and
- the supplied candidate-deck hypothesis.

Intent labels, policy-implementation labels, split labels, target returns,
target standard errors, selected hidden truth, raw hidden-zone card identities,
post-branch private state, and future continuation events are excluded from
model input. The action advantage remains exactly the centered, fixed causal
public branch heuristic from Phase 6AJ; no learned advantage is fitted.

## Frozen ridge selection

The scalar root target is the mean of the five empirical plan values. The
candidate model is a deterministic ridge regression with an unpenalized
intercept and named normalization fitted inside each training fold. The grid is:

- target form: direct root mean or residual from the pre-root heuristic; and
- penalty per training root: `0.25`, `0.50`, `1.0`, `2.0`, `3.0`, `5.0`.

The five development folds are complete policy/source groups:

1. Phase 6AI `canonical`;
2. Phase 6AI `engine_first`;
3. Phase 6AI `hand_first`;
4. Phase 6AI `action_first`; and
5. Phase 6AJ `action_first`.

For each candidate, every fold is predicted by a model fitted without that
fold. The frozen selection order minimizes:

1. worst-fold point-MSE ratio to the best of zero, pre-root heuristic, and
   standalone branch-heuristic baselines;
2. mean fold ratio;
3. mean fold MSE;
4. residual form before direct form; and
5. smaller penalty per root.

After selection, normalization and coefficients are refitted once on all 216
development roots. The prediction for root `s` and plan `a` is:

`clip(root_mean_ridge(s) + branch_heuristic(s,a)
      - mean_a branch_heuristic(s,a), -1, 1)`.

The selected configuration, all fold metrics, source fingerprints, root
ledger, normalizer, and coefficients must be serialized before collection.

## Fresh sequential panels

The official runtime is unseeded, but all abstraction-side allocation and
continuation sampling uses frozen seeds and namespaces:

- calibration: seed `6053`, namespace `phase6ak-calibration`; and
- validation: seed `6054`, namespace `phase6ak-validation`.

Each panel contains exactly one fresh `action_first` root for every cross of
the six candidate lists and six declared intent families: 36 complete matches,
72 supplied-allocation surfaces, 180 root/action targets, 360 immediate public
branch observations, and 2,880 terminal native continuations.

The model is frozen before calibration collection. The calibration panel then
constructs one global finite-sample 90% conformal radius from:

`abs(prediction - empirical_mean) + 1.645 * target_standard_error`.

The calibration artifact and its 36-root ledger must be serialized before any
validation worker can start. The validation panel is opened exactly once after
that bound is frozen. All 72 new roots must be mutually distinct and disjoint
from all 216 development roots.

## Frozen admission gates

The fresh validation panel must satisfy every point gate:

1. point MSE is lower than zero, the pre-root heuristic, the standalone causal
   branch heuristic, and the frozen Phase 6AJ hybrid evaluated on the same
   roots;
2. action-contrast MSE is at least 2% lower than zero contrast;
3. action-contrast MSE is no more than 1% worse than the standalone branch
   heuristic;
4. a deterministic 10,000-resample root bootstrap with seed `6055` has a
   strictly positive 90% lower bound for zero-minus-model contrast error;
5. maximum target standard error is at most `0.30`; and
6. model, calibration, and validation root ledgers are mutually disjoint in
   the roles declared above.

Conditional expected-mean admission additionally requires:

- overall mean-CI coverage at least `0.90`;
- mean clipped interval width at most `1.75`; and
- every action, candidate, and declared-intent slice coverage at least `0.75`.

Selected-action value/regret and pairwise concordance remain diagnostics. No
metric may be changed after validation is opened.

## Observed result

The model was serialized before collection. Cross-validation selected the
`root_heuristic_residual` form at penalty-per-root `5.0`; its worst-fold and
mean normalized MSE ratios were `0.754066` and `0.566429`. The calibration
panel was then completed and its radius frozen before any validation root was
collected. Two calibration roots required the preregistered second capture
attempt; all 36 validation roots succeeded on attempt one.

The untouched validation panel gives:

- ridge MSE `0.398773`, versus frozen Phase 6AJ `0.503581`, zero `0.683333`,
  pre-root heuristic `0.661293`, and standalone branch heuristic `0.663245`;
- ridge action-contrast MSE `0.027799`, versus zero `0.029653` and branch
  heuristic `0.027799`;
- 90% root-bootstrap lower bound `0.000442` for zero-minus-ridge contrast
  error;
- maximum target standard error `0.258199`;
- overall mean-CI coverage `0.916667` and mean clipped width `1.676068`; and
- minimum declared-slice coverage `0.666667`, occurring in the
  `board_denial` intent slice.

Thus every point, contrast, bootstrap, sampling-precision, and collection-
holdout ledger gate passes. The point model is a validated scoped diagnostic.
The interval gate fails solely because `0.666667 < 0.75`; conditional expected-
mean admission is rejected. The serialized field
`collection_holdout_gate_passed` refers to the preregistered fresh-root and
freeze-order ledger. It is not an implementation-heldout claim.

## Provenance and fail-closed boundary

Complete matches, legal selections, retained engine snapshots, exact replay,
immediate branches, and terminal descendants are genuine official-engine
execution. Candidate/intention policies, hidden-allocation sampling, the
five-plan boundary, tactical executor, continuation policy, public branch
heuristic, ridge model, bootstrap, and conformal intervals are abstractions.

Regardless of outcome, Phase 6AK keeps false:

- official hidden-allocation prior;
- human latent-intent calibration;
- opponent counterfactual-value compatibility;
- equilibrium convergence;
- safe-resolver enablement;
- opponent-exploitation enablement;
- official strategy control;
- Kaggle submission; and
- superhuman or other playing-strength claims.

## Frozen artifact ledger

Model fingerprint:
`5f3f73025251fde4d5e1dc155f604aa98856951030567e8baa13d2f1e999350e`

Calibration panel / calibration fingerprints:
`1c200b91807c1276d2a25de45aa4eb78bc3a66857ee26b30b3cb779353730ae6` /
`0904af96f6129588ba685d0e5aad7b25d6c5a18dc75d34ca9182efa3ab6b4229`

Validation panel / evaluation / evidence fingerprints:
`39bdc262987922fa33918fa217777d2ef2f5c1df634733c25e41692e0c14a451` /
`34bb66bcf50a0e8f122a9e289a92fb764974bf9034fbeac69fbce2ef2d76aba0` /
`edc94664e746f708ebbebf5805d9e4b3547ac3a64c58ba3e5c92db42b2bfe471`

| Artifact | SHA-256 |
|---|---|
| `expanded_root_mean_model.json` | `30d7c6c6bdc27f1bb64dff611491f06f97fe065bf7660cb795053a6fe2896857` |
| `calibration_panel.json` | `73ca8d95859f12516220980ad1f94d1411c6b7ca7ba4b8a3e127d0121c0aea2c` |
| `expanded_root_mean_calibration.json` | `484764d8ece7fb0546d935393defc1b9d8cdbc372b72592e59bd35b6afda1029` |
| `validation_panel.json` | `e75f420b2a5e4109dc5017a3ca8cbcc5ee195d07240be1530648039946dc58f7` |
| `expanded_root_mean_evaluation.json` | `288f2635f086dc2dce7acf4c2d2e3971befce378b4f0d83cb1ae5ebdb4273dbe` |
| `phase6ak_evidence.json` | `db88fd77a31e6785346c96e0c27da446395c5d3c20a6389366f19caf3ddbe38c` |
| `phase6ak_evidence.md` | `51b9ad86f800e3d6a1520e593751d44ce3befc93ddaca1d18ab507d52cf5d96c` |

The official runtime is unseeded, so the live panel is not claimed byte-
deterministic. The frozen artifacts parse and round-trip exactly, and the
saved ledgers—not re-execution—are the durable evidence for this run.

Final verification passed 205 tests and resolved all 915 public exports with
no duplicates. A clean PEP 517 wheel installed into a fresh environment; its
smoke command and `rocketbot-phase6-expanded-root-mean` entry point passed. The
wheel is 682,074 bytes with SHA-256
`a8b719fab3d45c3bd8eee4a98c7313de51a026f8c175fc8377f1647aff8dfb22`.
A second evaluation from the frozen saved panels changed zero of seven Phase
6AK artifact hashes.
