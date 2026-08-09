# N20 Trust-Region Projection Protocol

Status: **preregistered before projection evaluation**  
Protocol: `rocket-n20-trust-region-projection/v1`

## Rationale and separation from N19

N19 is rejected.  Its sole eligible terminal model learned the frozen fresh
leader validation/test actions by +8.12/+8.70 percentage points, but preserved
only about 97% of the live policy's top choices on exact-Rocket observations,
below the locked 99.5% gate.  It has no match or deployment authority.

N20 asks one new, bounded question: does the straight source-to-terminal weight
direction contain a smaller point that keeps a preregistered teacher gain while
restoring all fresh and historical exact-Rocket retention gates?  N20 is not a
checkpoint rescue.  It uses only the frozen live source and frozen epoch-4
terminal, a fixed scalar grid, and a fixed largest-safe-step rule.

## Frozen sources

- live source model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- rejected N19 terminal model SHA-256:
  `b0f0e72f7eafbf2bd5e0e4eca1920398534ddef516b57750a3c1f58dbb7d467f`;
- N19 training report SHA-256:
  `ed88d68c63988e79b74e0c33966bcdfe219cabfb78d9b0fe19f1482df7ebbe43`;
- N19 result SHA-256:
  `cfb2560fca269d969a9490453db28b7f7a3bf6598e96864cfd886ecc1962f227`;
- N19 corpus audit SHA-256:
  `d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

The source and terminal share all 59 state keys and tensor shapes.  Exactly the
preregistered final-block/action-head 14 tensors differ; the other 45 tensors
are byte-exact.

## Fixed projection family

For each changed tensor and scalar `alpha`, N20 constructs:

`projected = live + alpha * (N19_terminal - live)`.

Frozen tensors are copied byte-exact from live.  The only eligible grid is:

`{0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875}`.

`alpha=0` and `alpha=1` are recomputed as diagnostic endpoints only.  They
cannot be selected.  No nonlinear interpolation, tensor-specific scalar,
logit mixing, checkpoint mixing, seed, retraining, or additional grid point is
allowed.

## Evaluation support

Every grid point is evaluated on:

1. the unchanged N19 teacher train/validation/test examples;
2. the unchanged disjoint fresh exact-Rocket retention train/validation/test
   examples;
3. all 639 established N6R exact-feature examples, of which the 469 unchanged
   anchors are used for retention;
4. all 6,753 established N9 exact-Rocket held-out trajectory examples, retaining
   the two frozen opponent groups separately.

Base-policy probabilities and top choices are always recomputed from the exact
live model.  N6/N9 candidate metrics use full relational forward passes because
N20 changes the complete final transformer block; cached norm2-only shortcuts
are forbidden.

No official-engine match outcome, N18 arm outcome, reward label, opponent
private card, or search value enters selection.

## Per-grid-point gates

A grid point is safe only if every gate passes:

1. model configuration, keys, tensor shapes, finite values, serialization, and
   exact deck are valid;
2. only the 14 frozen N19 direction tensors differ from live;
3. teacher validation top-1 is at least +2.0 percentage points above live and
   validation loss is lower;
4. teacher test top-1 is at least +1.0 percentage point above live and test loss
   is lower;
5. base-policy top-1 agreement is at least 99.5% on each fresh retention split;
6. mean forward KL from live is at most 0.01 on each fresh retention split;
7. recorded-action retention validation and test top-1 are each no more than
   0.5 percentage points below live;
8. N6 unchanged-anchor top-1 retention is at least 99.0% and mean KL is at most
   0.05;
9. N9 top-1 retention is at least 99.0%, every frozen N9 group is at least
   98.5%, and mean KL is at most 0.02;
10. no extraction, legality, feature, full-forward, finite-value, or
    serialization error occurs.

## Sole selection rule

If no eligible grid point passes, N20 ends offline.  If one or more pass, the
sole selected candidate is the numerically largest passing `alpha`.  This is a
predeclared trust-region rule: maximize movement along the learned direction
subject to every independent safety and teacher gate.  Teacher score does not
break ties or choose among passing points.

Only the selected model is serialized as a candidate.  All other projections
are metrics only and have no downstream authority.

Offline passage permits only a separately locked both-seat native-engine
development panel against the live control.  It grants no confirmation,
packaging, submission, rank, strength, superhuman, or deployment claim.  The
current live bot remains unchanged.
