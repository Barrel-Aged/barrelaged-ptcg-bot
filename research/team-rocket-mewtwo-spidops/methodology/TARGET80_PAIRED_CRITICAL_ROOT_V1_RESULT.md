# Target-80 paired critical-root V1 result

Completed: 2026-08-05  
Decision: **closed at the preregistered stability gate; no learner, runtime,
development panel, package, or submission**

## Valid evidence

- The frozen 60-card substrate contains zero Wobbuffet.
- Both Marnie seat shards completed: 8 roots when Rocket started first and 8
  roots when Rocket started second.
- All 16 roots reproduced the guarded source action.
- All 256 required particle/coin matched sets completed, with every candidate
  continued through genuine official-engine transitions to terminal state.
- There were zero rollout errors, illegal branches, incomplete roots,
  visualizer reads, realized hidden-hand reads, or serialized hidden
  allocations.
- First-seat output SHA-256:
  `0ce1160b3b388cb8dcaf81b58bc9c8b866219797b27785e8f005bfc7b5d37329`.
- Second-seat output SHA-256:
  `5df91adfe1d75f989d6edee6ec67e5cbc7cc7bd3ea4e2910a85d89ec3f135cf3`.

The immutable analysis is
`work/target80_paired_critical_root_v1/analysis.json`.

## Locked gate result

| Gate | Result | Pass |
|---|---:|:---:|
| Integrity-complete roots | 16/16 | Yes |
| Particle-half ranking stability | 11/16 (68.75%) | No; required 12/16 |
| Coin-stream ranking stability | 13/16 (81.25%) | Yes |
| Stable non-baseline changes | 1/16 (6.25%) | No; required at least 3 |
| Stable change in each seat | second seat only | No |
| Mean delta over stable changes | +0.25 | Yes; required +0.125 |
| Criticality categories represented | 6 | Yes |

The sole stable change occurred from the second seat. Several other roots had a
positive full-sample alternative, but their particle halves disagreed. Many
candidate actions were exact terminal-value ties. The protocol therefore
records `passes_pilot: false` and `authority: none; close exact pilot`.

## Interpretation

Direct branching fixed the causal-identification problem of the randomized
terminal-outcome learner, but full-game win/loss remains too sparse and
particle-sensitive at this scale. The failure is not an engine, legality,
serialization, or hidden-information boundary failure. It is a target-density
and ranking-stability failure.

Do not increase particles, loosen the 75% threshold, select the one favorable
root as a rule, or distill this exact panel. The next materially different
experiment should retain paired official-engine branches but score a fixed,
short public horizon using the six strategic dimensions already defined by the
project: setup, tempo, resource efficiency, prize control, board control, and
optionality. Terminal outcomes reached inside that horizon should dominate the
dense score. That experiment requires its own prospective protocol and fresh
source roots.

The live V5 arena identity and every submitted byte remain unchanged.
