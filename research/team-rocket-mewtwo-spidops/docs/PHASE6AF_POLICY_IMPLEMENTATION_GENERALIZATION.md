# Phase 6AF: policy-implementation-heldout official intent families

Phase 6AF tests whether the controlled intent component from Phase 6AE transfers
across opponent-policy implementations rather than memorizing one exact action
priority table. It passes this implementation-heldout scope. It still does not
establish human latent intent, a complete public belief, counterfactual values,
safe resolving, exploitation, strategy control, Kaggle performance, or
superhuman play.

## Result

The untouched validation implementation reaches intent top-1 accuracy
`0.826389`, intent ECE `0.090055`, joint candidate-intent ECE `0.091196`, and
resource ECE `0.006552`. Every intent-family slice exceeds the preregistered
`0.50` top-1 floor; the minimum is lock/disruption at `0.583333`. The
implementation-heldout joint candidate / controlled-intent / hidden-resource
range therefore passes in scope.

## Four policy implementations

All implementations retain the six Phase 6AE intent-specific action budgets
and strategic ordering. They change how equal intent families prefer engine,
hand, or immediate-action operations:

| Role | Implementation | Collection ordinals | Use |
|---|---|---:|---|
| Training | `canonical` | 0-1 | Model fitting plus leave-one-policy-out selection |
| Training | `engine_first` | 2-3 | Model fitting plus leave-one-policy-out selection |
| Calibration | `hand_first` | 0-3 | Hyperparameter and confidence selection only |
| Validation | `action_first` | 0-3 | Untouched final evaluation only |

Player 0 remains on the same tempo policy across every intent and
implementation. Player 1 alone varies. The policy-implementation identifier is
supervision/provenance only and never enters model features.

## Collection and split discipline

- Six exact candidate deck lists.
- Six controlled intent families.
- Four roots per candidate/intent in each split.
- `144/144/144` train, calibration, and validation roots; 432 total.
- 32 public-consistent allocations per candidate/root.
- `432 x 6 x 32 = 82,944` candidate-conditioned particles.
- Player 1's second distinct MAIN turn is the root boundary.
- 11,105 genuine official selections and 1,188 public player-1 MAIN actions.

The two training implementations are cross-fitted by leaving one complete
implementation out. For every L2/temperature/uniform-mixture configuration,
selection records the worst metrics across:

1. canonical predictions from a model fitted only on `engine_first` roots;
2. `engine_first` predictions from a model fitted only on canonical roots; and
3. `hand_first` calibration predictions from a model fitted on both training
   implementations.

Selection maximizes the worst top-1 accuracy, then minimizes worst ECE, NLL,
and Brier score. The final model is fitted on both training implementations.
No `action_first` validation root, label, policy identifier, or metric is read
during fitting or selection. The selected configuration is L2 `0.1`,
temperature `0.03`, and uniform mixture `0.1`.

## Public-input and leakage boundary

The intent component consumes only the same Phase 6AE public representation:
allow-listed player-0-oriented root scalars plus counts, positions, transitions,
and fixed sequence slots for coarse public player-1 MAIN action types. It does
not consume either private hand, raw card IDs, raw selection indices, candidate
labels, intent labels, policy-implementation IDs, transient resource truth, or
player-0 action semantics.

Transient official visualizer truth supplies hidden hand/Prize/deck resource
labels only for offline evaluation. The frozen Phase 6AD candidate/resource
model is multiplied by the policy-generalized intent distribution to emit 36
normalized `RangeHypothesis` entries.

## Genuine execution versus abstraction

| Component | Status |
|---|---|
| Battle starts, prompts, legal selections, public transitions, and recorded action types | Genuine official `cg` execution |
| Transient realized hidden-resource samples | Genuine official visualizer state used only for offline labels |
| Observer adapter and public feature extraction | Research adapter |
| Four policy implementations and six intent-family labels | Controlled abstractions, not human truth |
| Cross-policy linear model and robust calibration | Offline research model |
| Candidate particles, 96 resource buckets, and candidate-intent product | Explicit abstractions |
| Chance reach, values, bounds, resolving, exploitation, and control | Unavailable and machine-disabled |

The official runtime remains unseeded. The checked fingerprints and telemetry
describe this run; recollection is not claimed byte-deterministic.

## Validation metrics

| Metric | Result |
|---|---:|
| Candidate NLL / ECE | `0.001150` / `0.002004` |
| Intent NLL / top-1 / ECE | `0.773500` / `0.826389` / `0.090055` |
| Joint NLL / top-1 / ECE | `0.774649` / `0.826389` / `0.091196` |
| Resource presence Brier / ECE | `0.042537` / `0.006552` |

Intent-family validation slices:

| Intent family | Roots | Top-1 | NLL |
|---|---:|---:|---:|
| Setup | 24 | `0.791667` | `0.921872` |
| Prize race | 24 | `0.750000` | `1.033465` |
| Lock/disruption | 24 | `0.583333` | `1.756734` |
| Board denial | 24 | `1.000000` | `0.087015` |
| Resource conservation | 24 | `1.000000` | `0.087011` |
| Combo assembly | 24 | `0.833333` | `0.754900` |

## Checked fingerprints

- Dataset:
  `cabc4bd77cc300dec2c1246c3aab27c80e13660579c0c685ae14d5ca2ac958f0`.
- Model:
  `b8c55327dcbe69f0a68e315f433140e6334723c4573281919f9a35e326b4ec5e`.
- Calibration:
  `5b91ee0ed27c5f8e0144bd5f5197915673d33c9ce4c8df1a10902c1d864bc5c2`.
- Evidence:
  `45d5d9df545ebba82066dcdf1214493c7d9ddb0ea453612f2007480a6a719c35`.

## Claim boundary and next dependency

Phase 6AF proves transfer only among four authored deterministic policy
implementations. It does not prove that human or Kaggle opponents express these
six labels, that the catalog covers unmodeled behavior, or that the product
range is a complete posterior. In particular, absolute official chance reach
is still undisclosed and there are no calibrated hypothesis-conditioned
counterfactual values behind these roots.

Phase 6AG follows this result with 288 complete official games. Its neural
terminal-outcome point leaf beats both point baselines on the untouched fourth
implementation, but its 90% intervals fail coverage and width. That result does
not provide hypothesis-conditioned expected or counterfactual values.

The next dependency is a fresh same-root collection that binds this
implementation-heldout range to repeated independent hidden completions and
complete opponent action-value surfaces. It must estimate conditional means
and calibrate collection-heldout value intervals before any
`CounterfactualValueBoundProvider` or resolver is enabled. Until then, safety,
exploitation, official control, and submission remain strictly false.

## Reproduce

```powershell
python scripts/phase6_official_policy_generalization.py
python -m unittest tests.test_official_policy_generalization -v
```

The collection writes under
`artifacts/phase6_official_policy_generalization/`. It makes no network request
and does not submit to Kaggle.
