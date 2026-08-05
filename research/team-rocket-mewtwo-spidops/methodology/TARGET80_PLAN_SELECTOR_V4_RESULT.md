# Target-80 state-conditioned plan selector V4 result

Completed: 2026-08-05  
Decision: **failed; exact V4 learner closed with no test or deployment authority**

## Integrity

- 160/160 training roots and 48/48 held-out validation roots completed.
- 3,328/3,328 matched particle/coin sets completed across 9,984 official-engine
  plan-line continuations.
- All six preregistered shards matched the frozen collector, identities,
  parameters, roots, seats, and seeds.
- Zero rollout errors, incomplete roots, guarded-baseline mismatches,
  opponent actions inside the scoring horizon, visualizer/realized-hidden reads,
  or raw hidden allocations.
- Frozen trainer result SHA-256:
  `bb83d36a4444fef1ebb422293184b5a04498d1e9b46989560eeac0315362a9cb`.

The training labels were 122 baseline, 36 `engine_resilience`, and two
`tempo_conversion`. The held-out labels were 32 baseline, 15
`engine_resilience`, and one `tempo_conversion`.

## Locked validation result

No configuration/threshold passed every gate. The safest and strongest setting
was the conservative network at threshold 0.85:

| Metric | Result | Locked gate | Pass |
|---|---:|---:|:---:|
| Change rate | 3/48 (6.25%) | 5%-30% | Yes |
| Exact plan precision | 2/3 (66.7%) | at least 65% | Yes |
| Exact plan recall | 2/16 (12.5%) | at least 20% | **No** |
| Baseline retention | 31/32 (96.9%) | at least 90% | Yes |
| Mean paired gain, all roots | +0.0127 | at least +0.04 | **No** |
| Mean paired gain, changed roots | +0.2026 | descriptive | - |
| Negative changed-root rate | 0/3 | at most 15% | Yes |
| Start-order mean gain | first +0.0006; second +0.0248 | both nonnegative | Yes |

Lower thresholds increased false changes and negative paired value. The other
two fixed architectures topped out at 50% exact-plan precision and +0.0028
mean validation gain. The immutable result records `passes_validation: false`,
`selected: null`, and `authority: none; close exact V4 learner` at
`artifacts/target80_plan_selector_v4/training_result.json`.

## Interpretation and next route

The paired teacher signal is real but sparse. The full 834-feature selector
found a very conservative high-value subset, yet did not generalize broadly
enough to justify a fresh test, full-match integration, packaging, or arena
submission. V4 therefore closes without threshold relaxation, architecture
retuning, or test collection.

The defensible next research route is a new compact selector centered on
`engine_resilience`, which accounts for 51 of 54 non-baseline labels across the
development corpus. V4's now-open development data may be used to identify a
small interpretable public-state feature set, but any successor must freeze
before a fresh game-disjoint test and must still pass full official-engine
match development and untouched confirmation.

The live V5 identity remains unchanged. No successor was submitted.
