# N9 Balanced Exact-Trajectory Robust AWR Result

Status: **rejected at the locked development gate; live bot unchanged**  
Protocol: `rocket-balanced-exact-pcgrad-awr/v1`  
Completed: 2026-07-30

## What was tested

N9 collected 1,200 fresh, complete official-engine trajectories from the
byte-frozen live Team Rocket policy, balanced equally between
`AshMarnieGeneralV110` and `HelbrosMarnieGeneralRecoveryV4`. Exact replay
proved parity for all 72,499 focal decisions, including 69,205 nontrivial
model action steps, with zero mismatches or excluded games.

Only `logit_head.weight` and `logit_head.bias` (385 parameters total) were
trained. Opponent-balanced game-level advantage weights and symmetric PCGrad
produced four fixed epochs. Opponent identity was used only to balance offline
gradients and was never available to the acting policy. All five arms passed
packaged-runtime parity before the strength panel.

The locked panel then completed 2,400 fresh second-seat official-engine games:
240 games per arm/opponent cell, zero fallbacks, and no correctness failures.

## Locked result

| Arm | V110 win rate | Change | Recovery V4 win rate | Change | Stratified change | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|---:|
| Control | 42.08% | — | 33.75% | — | — | — |
| Epoch 1 | 41.25% | -0.83 pp | 40.00% | +6.25 pp | +2.71 pp | [-3.54, +8.75] pp |
| Epoch 2 | 40.00% | -2.08 pp | 35.83% | +2.08 pp | +0.00 pp | [-6.25, +6.25] pp |
| Epoch 3 | 40.83% | -1.25 pp | 40.42% | +6.67 pp | +2.71 pp | [-3.54, +8.75] pp |
| Epoch 4 | 42.50% | +0.42 pp | 44.17% | +10.42 pp | +5.42 pp | [-0.83, +11.67] pp |

The preregistered gate required at least +2.0 points stratified, a strictly
positive change against each opponent, and a bootstrap lower bound greater
than -0.5 points. Epoch 4 passed both point-estimate gates but missed the
uncertainty gate by 0.33 points. Therefore no epoch advances under N9, and no
N9 checkpoint has validation, packaging, or deployment authority.

## Interpretation and next step

N9 provides the strongest development signal in this recovery campaign so
far. Unlike N7, epoch 4 did not trade away V110 strength, and its Recovery V4
gain was large. The remaining uncertainty is sampling uncertainty, not a
correctness, parity, legality, or opponent-leakage failure. The policy update
also moved only 385 action-head parameters, making it a narrow candidate for
independent confirmation.

The scientifically valid next experiment is a new, preregistered,
candidate-only confirmation using untouched seeds and enough games to absorb
the fact that epoch 4 was selected from four development checkpoints. It must
not reuse N9 development games or silently reinterpret N9 as a pass. A
confirmed candidate would still require broad matchup and seat regression,
packaged archive audit, and live evidence before any strength claim.

Immutable analysis:
`artifacts/team_rocket_recovery/n9_balanced_pcgrad_awr/development_panel/analysis.json`,
SHA-256
`b0bc2e5f6e063476101e68281162fc6ff7f6d1c40c4065b583f6e20f5c4d4522`.

Frozen epoch-4 model SHA-256:
`bdafc1ce36f7e572554930a1a9b1beff541a351ece09f934d0416b4ce269a020`.
