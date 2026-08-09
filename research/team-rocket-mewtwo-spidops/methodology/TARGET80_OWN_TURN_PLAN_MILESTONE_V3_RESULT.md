# Target-80 own-turn plan milestone V3 result

Completed: 2026-08-05  
Decision: **passed for expanded selector-data collection only**

## Integrity

- 16/16 roots completed across both Marnie seats.
- 256/256 matched particle/coin sets and 768 persistent plan lines completed.
- No opponent action entered the end-of-own-turn scoring horizon.
- Zero rollout errors, illegal branches, hidden-hand leaks, visualizer reads,
  raw hidden allocations, fallback exceptions, or guarded-baseline mismatches.
- First-seat output SHA-256:
  `c2da32a24dbd8684a59b9b15a81e2e7c297b54174aaa1cb58a6755f3e54483f3`.
- Second-seat output SHA-256:
  `8410c3c867745d010d08fd1d5574ccb917d18a445196148e2ca02acf4375e1bf`.

## Locked gates

| Gate | Result | Pass |
|---|---:|:---:|
| Integrity-complete roots | 16/16 | Yes |
| Particle-half ranking stability | 12/16 (75.0%) | Yes |
| Coin-stream ranking stability | 12/16 (75.0%) | Yes |
| Stable non-baseline plans | 5/16 (31.25%) | Yes |
| Stable changes by seat | 1 first, 4 second | Yes |
| Mean dense gain over stable changes | +0.310 | Yes |
| Criticality categories represented | 6 | Yes |

Four stable changes selected `engine_resilience`; one selected
`tempo_conversion`. The immutable analyzer records `passes_pilot: true` and
`authority: expanded own-turn plan corpus only` at
`work/target80_own_turn_plan_milestone_v3/analysis.json`.

## Claim boundary and next action

This result establishes that the project can obtain stable, causally paired
strategic-line labels at the controllable end-of-own-turn boundary. It does not
show that those labels improve match win rate and does not authorize a runtime,
development panel, package, arena submission, or strength claim.

The authorized next step is a larger game-disjoint corpus from fresh roots,
followed by a conservative state-conditioned plan selector. The selector must
retain the exact baseline on uncertain examples, pass held-out plan-value and
seat gates, and then beat baseline in full official-engine matches before any
confirmation or packaging.

The live V5 identity remains unchanged.
