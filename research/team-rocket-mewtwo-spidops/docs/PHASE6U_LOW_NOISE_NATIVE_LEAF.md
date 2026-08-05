# Phase 6U: low-noise allocation-heldout native leaf diagnostic

Phase 6U isolates sampling noise from leaf-model bias. It increases each cutoff
target from three to ten independent terminal continuations, uses hidden-zone
allocation particles rather than order-only particles, and selects among three
existing value baselines on calibration particles before touching the held-out
validation allocations.

## Dataset and split

One native root supplies four allocation variants for each of the six deck
candidates:

- variants 0-2 form 18 calibration particles;
- variant 3 forms six validation particles; and
- every particle reaches one two-macro-step cutoff and then branches into ten
  independent fixed-profile terminal continuations.

The checked dataset therefore contains 24 cutoff examples, 240 genuine native
terminal continuations, and 29,690 native transitions. It persists hashed/
bucketed public and actor-visible private features but no raw private card IDs.
Candidate identity is evaluation-only and is not a model input.

## Pre-existing model comparison

No validation row is used to fit or select a model. The calibration split
compares:

1. the split-safe Phase 6I logged-trajectory outcome head, with the causal
   revealed-card feature baseline established by Phase 6T;
2. the Phase 6K tactical candidate's confidence-zero scalar; and
3. a zero-value baseline.

The outcome head predicts the acting player's utility and is sign-corrected to
player 0. Its actor-visible private input is legitimate for the acting
information set but is one reason this diagnostic is not a public
counterfactual-value provider.

| Model | Calibration MAE | Calibration RMSE | Validation MAE | Validation RMSE |
|---|---:|---:|---:|---:|
| Phase 6I logged value | 1.131540 | 1.238751 | 1.156624 | 1.230277 |
| Phase 6K tactical value | 0.703028 | 0.808935 | 0.455094 | 0.607362 |
| Zero baseline | **0.444444** | **0.533333** | **0.366667** | **0.454606** |

Calibration selects the zero baseline. This is evidence that neither learned
head transfers as a useful native cutoff value under the current representation
and objective.

## Precision and interval result

Ten bounded `[-1, 1]` returns reduce the maximum observed rollout standard
error to `0.333333`, below the predeclared `0.35` threshold. Thus Phase 6S's
sampling-precision failure is resolved for this suite.

The selected model's 90% split-conformal radius is nevertheless `1.000000001`.
After clipping to the legal utility range:

| Metric | Result |
|---|---:|
| Validation coverage | 1.000 |
| Validation MAE | 0.366667 |
| Validation RMSE | 0.454606 |
| Mean interval width | 2.000000 |
| Informative interval fraction | 0.000 |
| Maximum rollout standard error | 0.333333 |

Coverage remains vacuous, so the empirical fixed-profile leaf gate fails its
width requirement. The blocker is now model bias/representation, not the
declared rollout-precision threshold.

## Genuine execution versus abstraction

| Component | Classification |
|---|---|
| Terminal descendant selections and results | Genuine licensed native-search execution |
| Hidden allocation particles | Supplied research particles |
| Cutoff and five-plan action space | Research abstraction |
| Tactical and plan policies | Research abstractions |
| Ten-rollout mean | Fixed-profile empirical target |
| Model selection and conformal interval | Research evaluation |
| Opponent counterfactual value | Not estimated |

## Fail-closed consequence

Phase 6U does not build counterfactual-value bounds and does not invoke a safe
resolver, exploitation overlay, official strategy controller, packager, or
Kaggle submission. Even a future informative fixed-profile interval would
still require a separate opponent counterfactual-value audit before safety
admission.

Phase 6V tests the first purpose-trained public scalar on a separate native
allocation family. It improves fresh point RMSE but still yields an almost
full-width conformal interval, so it is rejected. The next value step needs a
broader native public-belief value model with hypothesis-conditioned opponent
counterfactual outputs, trained across many independent roots and
allocation-disjoint targets. The Phase 6I behavior/outcome head, Phase 6K
tactical scalar, and Phase 6V ridge should remain initialization/ablation inputs,
not safety baselines.

## Durable evidence

Evidence fingerprint:
`52fb90a7960dc73887fc92c0268e8cf3bac05a1322c3dcbaebe4a27e8791aa04`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `native_low_noise_leaf_dataset.json` | 420,392 | `2f4ad548b4ea1a11254fdefae1c287772f219b09b1df8c316ba6d5990e05c384` |
| `native_leaf_model_calibration.json` | 2,303 | `64ba89587caea12546044ebe935ffb2775071de713256f5dac0839ce284f3cb2` |
| `phase6u_evidence.json` | 1,782 | `38dee9a2a063bd02a6d69930e73204938263838a8017452508904e24b648e2cb` |
| `phase6u_evidence.md` | 1,030 | `3873ee2cad4b04b2db8adbd32b4b8edf987cea7cef48d908a78b14d318e0281f` |

Reproduce with:

```powershell
python scripts/phase6_native_leaf_model.py
python -m unittest tests.test_native_leaf_model -v
```

The competition runtime is unseeded, so independent runs are not claimed
byte-deterministic. Fingerprints bind this observed run.
