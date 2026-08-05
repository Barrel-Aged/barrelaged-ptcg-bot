# N8B Recovery Logit-Delta Transfer Result

Status: **rejected at the locked development gate; live bot unchanged**  
Protocol: `rocket-recovery-logit-delta-transfer/v1`  
Completed: 2026-07-30

## What was tested

N8B isolated the two action-head tensors that distinguish
`HelbrosMarnieGeneralRecoveryV4` from its otherwise parameter-identical
`AshMarnieGeneralV110` sibling. The Recovery-minus-V110 delta was added to the
Rocket-adapted policy at fixed scales 0.5, 1.0, 1.5, and 2.0. No trajectory was
used for fitting and every non-logit-head policy tensor remained frozen.

The arm builder proved source hashes and tensor identity. All five arms then
matched the packaged greedy action path on 2,505/2,505 official-engine
decisions across 40 games. The locked panel completed 2,400 fresh second-seat
official-engine games—240 per arm/opponent cell—with zero fallbacks or
correctness failures.

## Locked result

| Arm | V110 win rate | Change | Recovery V4 win rate | Change | Stratified change | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|---:|
| Control | 45.00% | — | 39.17% | — | — | — |
| Delta 0.5 | 43.75% | -1.25 pp | 37.92% | -1.25 pp | -1.25 pp | [-7.50, +5.00] pp |
| Delta 1.0 | 45.42% | +0.42 pp | 41.25% | +2.08 pp | +1.25 pp | [-5.00, +7.50] pp |
| Delta 1.5 | 42.08% | -2.92 pp | 37.92% | -1.25 pp | -2.08 pp | [-8.33, +4.17] pp |
| Delta 2.0 | 45.00% | +0.00 pp | 40.42% | +1.25 pp | +0.63 pp | [-5.63, +6.88] pp |

No arm reached the preregistered +2.0-point stratified improvement and
bootstrap gates. `delta_100` was directionally positive against both opponents
but its +1.25-point aggregate estimate is too small and uncertain to justify
validation or deployment.

## Interpretation and next step

The clean source delta is compatible with both target policies, unlike N7's
V110 exploit, but it is not a large enough intervention by itself. This makes
blind scalar refinement unattractive: the observed response is non-monotone
and repeated line search on the same two opponents would overfit development
noise.

N9 should instead collect fresh, exact, balanced trajectories from the current
control against V110 and Recovery V4, preserve opponent identity as a public
training stratum, and perform game-balanced outcome-weighted regression on the
small action head. That directly addresses N7's distribution imbalance while
retaining N8B's finding that a minimal action-head change can improve Recovery
without sacrificing V110.

Artifact:
`artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_panel/analysis.json`.
No N8B artifact was validated, packaged, or submitted.

