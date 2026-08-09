# Phase 6AO: Selective Search Gate Recovery Protocol

Status: locked after the Phase 6AN untouched gate-concentration test failed and
before Phase 6AO gate metrics are computed.

## Failure being repaired

Phase 6AN's official counterfactual bounds passed, but its single hidden-state
gate logit did not concentrate available plan-search improvement on the
untouched roots. The submitted controller cannot use that failed gate. Phase
6AO reopens only the selective-search component; the trained blueprint,
beliefs, strategic heads, tactical policy, legality boundary, and admitted
bounds remain unchanged.

## Data boundary

- Development: all 180 Phase 6AH expected-value roots. The previous untouched
  partition is now explicitly development data because its failed gate result
  has been observed.
- Threshold calibration: the 36 root-disjoint Phase 6AL high-precision
  calibration roots.
- Final untouched admission: the 36 root-disjoint Phase 6AL high-precision
  validation roots. No Phase 6AO parameter, feature, or threshold is selected
  from these roots.

All three panels contain complete official source matches and repeated
licensed native continuations. Plan actions, hidden allocations, and the
frozen continuation policy remain research abstractions. Root fingerprints
must be mutually disjoint.

## Gate model

A separate two-layer gate consumes only values available to the deployed
controller: the original gate probability, uncertainty, plan and belief
entropy, plan-value spread, predicted best-plan advantage, strategic-head
disagreement, policy confidence, prize threshold, turn, option count, and
public board deltas. It is trained to rank the observed improvement available
from choosing the best of the five repeated-continuation plan actions instead
of the deployed blueprint mixture.

The architecture, seed, 16-unit hidden width, optimizer, and 2,000-step limit
are fixed in source. Only a gate threshold is selected on the calibration
panel. Eligible thresholds must trigger between 20% and 60% of calibration
roots. The selected threshold maximizes gated minus routine mean available
improvement, with the higher threshold breaking ties.

## Admission

The Phase 6AO gate is admitted only if, on the untouched validation panel:

- both gated and routine sets are nonempty;
- between 15% and 70% of roots trigger;
- gated mean available improvement exceeds routine mean by at least 0.02;
- gated mean available improvement exceeds the panel-wide mean; and
- runtime/training forward parity remains within `1e-6`.

If any gate fails, online re-solving remains disabled and the completion cycle
restarts. Passing this test shows selective concentration under this fixed
official panel; it is not a general theorem about search value.
