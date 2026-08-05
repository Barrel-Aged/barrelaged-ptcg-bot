# Target-80 compact engine-resilience rule V5 test result

Completed: 2026-08-05  
Decision: **failed; exact compact rule closed with no integration authority**

## Integrity

- 96/96 fresh roots completed across both Rocket start orders.
- 1,536/1,536 matched particle/coin sets and 4,608 official-engine plan lines
  completed.
- Zero rollout errors, incomplete roots, guarded-baseline mismatches, opponent
  actions inside the scoring horizon, visualizer/realized-hidden reads, or raw
  hidden allocations.
- First-start output SHA-256:
  `3103fb2e701347e7fb3341912e9a5a0019ed172f05880f2051962dcd1c1fa5f3`.
- Second-start output SHA-256:
  `ff05d1af35111ad0c21070298f4dbef5eced667c6166c889243e13df0dd32eff`.
- Locked analysis SHA-256:
  `134d9d01b70c7822462c08a4c9cb4092984010d48212bf713221fe5512422d40`.

## Frozen gates

| Gate | Fresh result | Required | Pass |
|---|---:|---:|:---:|
| Integrity | 96 roots / 1,536 sets | exact | Yes |
| Change rate | 14/96 (14.6%) | 5%-30% | Yes |
| Exact-plan precision | 7/14 (50.0%) | at least 65% | **No** |
| Exact-plan recall | 7/20 (35.0%) | at least 20% | Yes |
| Stable-baseline retention | 70/76 (92.1%) | at least 90% | Yes |
| Mean paired gain, all roots | +0.0368 | at least +0.04 | **No** |
| Mean paired gain, changed roots | +0.2525 | descriptive | - |
| Negative changed-root rate | 3/14 (21.4%) | at most 15% | **No** |
| Start-order mean gain | first +0.0375; second +0.0362 | both nonnegative | Yes |

The immutable analyzer records `passes_test: false` and
`authority: none; close compact engine rule V5` at
`work/target80_compact_engine_rule_v5_test/analysis.json`.

## Interpretation

The rule's continuous causal value generalized directionally: both start
orders were positive and changed-root gain remained about +0.25. It did not
generalize the stable-label precision or negative-tail control needed for safe
integration, and its all-root value narrowly missed the locked minimum.

Several nominal false positives had positive full-sample deltas but unstable
particle/coin rankings. That diagnostic does not override the frozen gates.
The exact broad rule is closed without post-test edits, full-match integration,
packaging, or submission. Any narrower action guard or different continuous-
value objective is a new hypothesis and requires a new prospective protocol
and fresh evidence.

The live V5 identity remains unchanged.
