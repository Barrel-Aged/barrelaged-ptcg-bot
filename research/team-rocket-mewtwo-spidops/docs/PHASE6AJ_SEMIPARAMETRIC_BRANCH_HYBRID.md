# Phase 6AJ preregistration: semiparametric branch-value hybrid

Status: **fresh action-sensitive point signal validated; conditional intervals
rejected; every downstream capability remains fail-closed**

## Question

Phase 6AI's grouped network transfers root-mean value but overwrites the useful
action ordering in its causal public branch baseline. Phase 6AJ tests a narrower
model that cannot do that: use a neural root mean and the centered, fixed causal
branch heuristic as the complete action advantage.

This is a fresh confirmation test for an action-sensitive conditional point
model. It is not an equilibrium, opponent-counterfactual-value, interval,
safe-resolving, deployment, leaderboard, or playing-strength claim.

## Frozen source fitting

Phase 6AJ consumes the frozen Phase 6AI dataset and selected network. It uses
only the selected network's scalar root-mean output; the learned Phase 6AI
advantage output is discarded. No Phase 6AI validation target is used for
selection or calibration.

For root `s` and plan `a`, the prediction is:

`clip(mean_blend * neural_mean(s) + (1 - mean_blend) * root_heuristic(s)
      + branch_scale * (branch_heuristic(s,a) - mean_a branch_heuristic(s,a)))`.

`hand_first` replica 0 selects from this fixed grid:

- root-mean blend: `0.25`, `0.50`, `0.75`, `1.0`; and
- branch-advantage scale: `0.50`, `0.75`, `1.0`, `1.25`.

Selection minimizes the worst normalized point and zero-contrast error. The
Phase 6AI `hand_first` replica 1 roots construct the mean-CI conformal radius.
All selected parameters, source fingerprints, and bounds are frozen before the
new panel is opened.

## Fresh validation panel

Phase 6AJ uses seed `6051` and a new `phase6aj` namespace. It collects exactly
36 complete official matches: one `action_first` match for every cross of the
six candidate lists and six declared intent families. Root selection, two
supplied public-consistent allocations, all five plans, and eight paired
terminal continuations per allocation/action are unchanged.

The validation total is 36 fresh roots, 72 supplied-allocation surfaces, 180
root/action targets, 360 immediate public branch observations, and 2,880
terminal native continuations. Every Phase 6AJ root must be identity-disjoint
from all Phase 6AH and Phase 6AI roots. No Phase 6AJ target may influence model
selection or interval construction.

## Frozen gates

The fresh validation result must satisfy all action-sensitive point gates:

1. point MSE is lower than zero, the pre-root heuristic, and the standalone
   causal branch heuristic;
2. contrast MSE is at least 2% lower than zero contrast;
3. a deterministic 10,000-resample root bootstrap has a strictly positive 90%
   lower bound for zero-minus-hybrid contrast-MSE improvement;
4. maximum target standard error is at most `0.30`; and
5. source fitting/calibration roots and all fresh validation roots are disjoint.

Conditional expected-mean admission additionally retains the Phase 6AI
interval gates: coverage at least `0.90`, mean width at most `1.75`, and every
action/candidate/intent slice coverage at least `0.75`.

Selected-action value/regret, pairwise concordance, and the standalone branch
heuristic's bootstrap are diagnostics. Equality with the branch heuristic's
action ranking is expected by construction and is not claimed as a learned
action advantage.

## Provenance and fail-closed boundary

Genuine official execution is limited to complete matches, legal selections,
retained snapshots, exact replay, immediate native branches, and terminal
native continuations. The supplied allocation law, five plans, tactical
executor, continuation blueprint, public branch heuristic, neural root mean,
hybrid, bootstrap, and intervals are abstractions.

Regardless of outcome, Phase 6AJ keeps false:

- official hidden-allocation prior;
- opponent counterfactual values;
- equilibrium convergence;
- safe-resolver enablement;
- opponent-exploitation enablement;
- official strategy control;
- Kaggle submission; and
- superhuman or other playing-strength claims.

## Frozen result

The model and bounds were serialized before the first validation root was
collected. Selection chose root-mean blend `0.5` and branch-advantage scale
`1.0`; the `hand_first` bounds split produced conformal radius
`1.3882648651606098`. The immutable fingerprints are:

- source Phase 6AI dataset:
  `8a56e5f44d9fe3c3c680c5a586e02292e01bbc65c14ed3aa653c355ab312f2a8`;
- source Phase 6AI model:
  `4c195b8dad7876ce03d2f433ba7d2393675d7fdcde9aedc9ea78e6e79ea18b7a`;
- source Phase 6AI calibration:
  `1295b226009eeb130c9390b73972ce83f48049e5823c6f03419c4789ac0bee29`;
- Phase 6AJ model:
  `1ff5ad615e502f299a222b735a83be08e3049127843d435998b124fdf6cd8374`;
- Phase 6AJ calibration:
  `c6e6cad008ee761856cc298ee80d69efeb7e2225bf62a14f5d49b9c9128869a6`;
- fresh validation dataset:
  `8b21619c5ae9be87ab22c574ddba9260629f4925a943fd937d49cd9d23a7dfdc`;
- evaluation:
  `ac0912bf58238d2a9f764d2a4e01a040cf6c56c7bae8313f8ffe884bfe2be9f8`;
  and
- evidence:
  `9d78e4364d60ecfb032febbdae0ef90c0909ef1c5babc6971fdfc94bf8b4d359`.

All semantic fingerprints above are enforced by the JSON artifact loaders.
The fresh panel contains
36 complete official matches, 3,840 legal live selections, 360 immediate native
branches, 2,880 terminal native continuations, and 117,341 descendant native
transitions. All 36 root identities are disjoint from the 180 Phase 6AH and 180
Phase 6AI roots. Thirty-five captures completed on attempt 1 and one completed
under the frozen deterministic retry allowance on attempt 2.

| Frozen validation metric | Result | Gate |
|---|---:|---:|
| Hybrid point MSE | `0.5828403435` | pass |
| Zero / root / standalone branch MSE | `0.7642361111` / `0.8914873649` / `0.8537162772` | hybrid lower |
| Hybrid / zero action-contrast MSE | `0.0228683927` / `0.0252777778` | pass |
| 90% bootstrap lower bound, zero minus hybrid contrast error | `0.0007529020` | pass |
| Maximum target SE | `0.2581988897` | pass |
| Overall coverage | `0.9055555556` | pass |
| Mean clipped interval width | `1.8397911182` | **fail** (`> 1.75`) |
| Minimum slice coverage | `0.8000000000` | pass |
| Implementation holdout | `true` | pass |

The preregistered action-sensitive point admission therefore passes. The model
selects a mean realized value of `-0.0763888889`, identical to the standalone
branch heuristic's ranking by construction, versus an oracle value of
`0.0104166667`; mean regret is `0.0868055556` and pairwise concordance is
`0.6056338028`. These are diagnostics, not playing-strength claims.

Conditional expected-mean admission fails solely because the intervals remain
too wide. The result does not provide opponent counterfactual values or justify
putting this point predictor into the blueprint, resolver, exploitation layer,
or official controller. The next admissible value experiment must reduce
collection-heldout interval width without using the Phase 6AJ targets for model
or calibration selection, while preserving this action-contrast result.

## Durable artifacts

| Artifact | Bytes | File SHA-256 |
|---|---:|---|
| `branch_hybrid_model.json` | 991,168 | `0ce6f275d1c47237ae00b377e6df00833c6cfb1c61a646ada6d24cd03370d1a5` |
| `branch_hybrid_calibration.json` | 3,944 | `59aeab88f8b2e34383c9b777fcbbf409ba7c98afa983eb6e88d0adc679185c44` |
| `branch_hybrid_validation_dataset.json` | 1,670,872 | `2ce7e02bb83a1762e88b70599c4e834db2f5367b025d8332f42c2d3038db1444` |
| `branch_hybrid_evaluation.json` | 8,929 | `acc849a6ba28277af2c33817713c50c7ed95259831f0abe2ee9a9ce4e254347d` |
| `phase6aj_evidence.json` | 1,953 | `93ce4a5b92cd80272591943fc952ac221fdb8fbef64a215925f2ad231cac712f` |
| `phase6aj_evidence.md` | 1,654 | `3d791003e1ba5689ff71995abf5e04e1151233145af59426e4fd326b70cdd390` |

Re-evaluating the frozen validation dataset changed none of these six files.
The unseeded native engine means a fresh recollection is not claimed to be
byte-identical.
