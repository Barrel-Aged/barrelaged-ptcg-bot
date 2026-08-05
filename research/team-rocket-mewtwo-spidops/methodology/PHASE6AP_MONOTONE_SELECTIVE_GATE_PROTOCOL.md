# Phase 6AP: Monotone Selective Gate Protocol

Status: locked after Phase 6AO failed and before the Phase 6AM public-range
validation panel is scored by this gate.

## Restart boundary

Phase 6AO overfit its calibration roots: it separated calibration benefit but
reversed on validation. Phase 6AP therefore discards that candidate gate. It
keeps the Phase 6AN controller and its independently admitted bounds, then fits
a deliberately lower-capacity monotone gate.

## Partitions

- Development: the 180 Phase 6AH roots plus the 36 Phase 6AL calibration
  roots. Both have now been used during gate development.
- Threshold selection: the 36 Phase 6AL validation roots. Their failed Phase
  6AO result is now development knowledge, so they are used only to choose the
  Phase 6AP operating point.
- Untouched admission: the 36 Phase 6AM public-range counterfactual-advantage
  validation roots. Their Phase 6AP gate scores and benefit concentration have
  not been computed before locking this document.

All root ID sets must be disjoint. The final panel's candidate summaries,
public belief, range weights, and supervision-only labels are excluded from the
gate. Only its repeated official continuation values are used after the gate
has produced scores.

## Frozen model

The gate is a single logistic unit with four inputs already available in the
submitted controller:

- value uncertainty, constrained to a nonnegative coefficient;
- opponent-archetype entropy, constrained nonnegative;
- disagreement among the six strategic heads, constrained nonnegative; and
- turn number, constrained nonpositive.

All other gate-feature coefficients are exactly zero. The inputs are
standardized using development roots only. Coefficients are fit for 3,000 full
batch Adam steps with seed 61,502, learning rate 0.02, an action-benefit
point-loss, pairwise ranking loss weight 0.5, and L2 weight 0.02. There is no
architecture or seed sweep.

The threshold must trigger 20% to 60% of selection roots and maximizes gated
minus routine mean available improvement; higher threshold breaks ties.

## Admission

On the untouched Phase 6AM panel, both partitions must be nonempty, 15% to 70%
of roots must trigger, gated mean available improvement must exceed routine by
at least 0.02, and gated mean must exceed the overall panel mean. Runtime parity
must remain within `1e-6`. Otherwise this candidate is rejected and the plan
restarts again.

Passing is evidence of selective concentration on one fresh official panel,
not a universal or equilibrium guarantee.
