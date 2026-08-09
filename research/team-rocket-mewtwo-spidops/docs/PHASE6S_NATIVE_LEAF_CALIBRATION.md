# Phase 6S: native leaf-continuation calibration diagnostic

Phase 6S tests whether the confidence-zero Phase 6K cutoff scalar introduced in
Phase 6R is close enough to repeated terminal continuations to justify a useful
leaf interval. It also measures the current static six-candidate public range
against the selected evaluation candidate. The answer is no, and the failure
is encoded as a durable, fail-closed artifact.

## Held-out particle construction

`permute_hidden_zone_completion` creates new particles by deterministically
permuting order within each already public-consistent hidden zone. Card counts
and allocation between deck, prize, hand, and Active are unchanged. This gives
unseen draw/order states without inventing cards or leaking them into policy
features.

Each of the six opponent candidates has three variants:

- variants 0 and 1 are calibration particles;
- variant 2 is a validation particle;
- the 12 calibration and six validation particle IDs are disjoint;
- all six candidates occur in both splits.

This validates only held-out within-zone order. It does not validate alternative
hidden-zone allocations, an independently sampled posterior, or a
history-conditioned belief model.

## Continuation dataset

For each particle, the fixed Phase 6P plan provider and Phase 6K tactical
provider sample an abstract path to a two-macro-step cutoff. Three independent
continuations then branch from that retained native state and run to a genuine
terminal result through the licensed search engine.

The checked-in dataset contains:

- 18 cutoff examples across six candidate strata;
- 12 calibration and six validation examples;
- 54 genuine terminal continuations;
- 6,240 native transitions;
- actor-visible hashed/bucketed public and private features;
- zero persisted raw hidden card IDs.

The selected candidate and terminal results live only in evaluation-target
fields. They never enter the policy feature payload.

## Leaf result

The calibration residual radius is `1.597915528`. Clipping the corresponding
intervals to the legal `[-1, 1]` utility range gives:

| Metric | Result |
|---|---:|
| Calibration coverage | 1.000 |
| Validation coverage | 1.000 |
| Validation MAE | 0.682560 |
| Validation RMSE | 0.725152 |
| Mean interval width | 2.000000 |
| Informative interval fraction | 0.000 |
| Maximum rollout standard error | 0.666667 |

Coverage is therefore vacuous: every validation interval is the full utility
range. The declared empirical rollout-leaf gate fails both the maximum interval
width (`1.25`) and maximum standard-error (`0.35`) requirements.

## Public-range result

The player-0 root range remains the static equal-weight completion prior. Across
all 18 evaluation particles it produces:

| Metric | Result |
|---|---:|
| NLL | 1.791759 |
| Uniform NLL | 1.791759 |
| Brier score | 0.833333 |
| Top-1 accuracy | 0.166667 |

It contains no information beyond the uniform six-candidate baseline. The
public-belief calibration prerequisite remains false.

## What is genuine and what is abstract

| Component | Classification |
|---|---|
| Terminal descendant selections and results | Genuine licensed native-engine execution |
| Hidden order particles | Supplied research particles |
| Five plan actions and tactical executor | Research abstraction |
| Cutoff state and leaf prediction | Research abstraction |
| Repeated fixed-profile return | Empirical rollout target, not an exact counterfactual value |
| Candidate truth label | Evaluation-only field |
| Public range | Static supplied prior, uncalibrated |

The dataset itself is not official-engine execution even though its descendant
targets come from genuine native transitions.

## Fail-closed consequence

Phase 6S builds no counterfactual bound and does not invoke a resolver,
exploitation overlay, official strategy controller, packager, or Kaggle
submission. It does not satisfy Phase 6Q's public-belief, held-out-allocation,
independent-counterfactual-value, model-capability, or equilibrium gates.

The next admissible step is to sample hidden-zone allocations rather than only
orders, train a history-conditioned public-range model on more match-disjoint
replays, and collect enough independent continuations to reduce target standard
error. Any future leaf interval must be informative on a held-out candidate and
particle suite before it can even enter a counterfactual-value audit.

Phase 6T implements the allocation sampler and tests the existing Phase 6I
public-only range head. That head is worse than uniform and the causal
revealed-card updater on held-out native allocation rows, so belief calibration
remains false. Lower-noise continuation targets and a new belief baseline are
still pending; see `docs/PHASE6T_NATIVE_ALLOCATION_RANGE.md`.

## Reproduce

```powershell
python scripts/phase6_native_leaf_calibration.py
python -m unittest tests.test_native_leaf_calibration -v
```

Artifacts are under `artifacts/phase6_native_leaf_calibration/`. The native
runtime is unseeded, so independent runs are not claimed byte-deterministic;
fingerprints bind this observed run.
