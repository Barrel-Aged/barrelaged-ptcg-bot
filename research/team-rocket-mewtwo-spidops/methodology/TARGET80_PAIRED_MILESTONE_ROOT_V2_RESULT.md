# Target-80 paired milestone-root V2 result

Completed: 2026-08-05  
Decision: **closed at stability; no selector, development panel, package, or
submission**

## Valid evidence

- 12/12 plan-level roots completed across both Marnie seats.
- All 192 particle/coin matched sets and 576 persistent-plan continuations
  reached their locked two-turn horizon or terminal state.
- Every strategic aggregate reproduced from the six serialized axes and the
  root-conditioned meta weights.
- There were zero rollout errors, illegal branches, hidden-hand leaks,
  visualizer reads, raw hidden allocations, or guarded-baseline mismatches.
- First-seat output SHA-256:
  `fbe8f3c140e967e5db8e469b2d938129c7b18de4cb3d8288917f7c7ba5e472a5`.
- Second-seat output SHA-256:
  `f4b208d1b33893cff4fe901c1a6eab36359374a72f44b8ac13523e002a108c23`.

## Locked result

| Gate | Result | Pass |
|---|---:|:---:|
| Integrity-complete roots | 12/12 | Yes |
| Particle-half ranking stability | 6/12 (50.0%) | No; required 9/12 |
| Coin-stream ranking stability | 5/12 (41.7%) | No; required 9/12 |
| Stable non-baseline plans | 4/12 (33.3%) | Yes |
| Stable change in each seat | 2 first, 2 second | Yes |
| Mean dense gain over stable changes | +0.222 | Yes |
| Criticality categories represented | 5 | Yes |

All four stable changes selected `engine_resilience`, but the protocol did not
permit extracting that favorable subset when overall ranking stability failed.
The immutable analysis records `passes_pilot: false` and
`authority: none; close exact V2 pilot` at
`work/target80_paired_milestone_root_v2/analysis.json`.

## Interpretation

Persistent plan lines made the target materially denser and produced coherent
positive examples in both seats, but including the opponent response through
Rocket's next turn introduced too much hidden-particle and coin-stream
variance. V2 therefore does not authorize a learned plan selector.

Do not loosen the 75% gates, increase particles on these inspected roots, or
promote `engine_resilience` directly. The next distinct diagnostic may retain
the same prospectively specified plan proposals while moving the milestone to
the end of Rocket's current turn, before the opponent acts. That measures the
complete controllable strategic line, removes opponent-response variance, and
still requires later full-game official-engine development and independent
confirmation before deployment.

The frozen Wobbuffet-free substrate and live V5 submission remain unchanged.
