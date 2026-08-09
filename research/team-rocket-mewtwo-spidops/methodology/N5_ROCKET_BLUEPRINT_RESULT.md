# N5 Rocket Blueprint Fine-Tune Result

## Outcome

N5 did **not** advance a candidate. No N5 model is eligible for sealed
validation, broader regression, packaging, or internal-arena submission.

This is development-selection evidence only. It is not validation evidence and
does not support an arena-strength claim.

## Locked campaign executed

- Frozen preregistration manifest SHA-256:
  `6300ffae4d6a73e59ce11695fa9196110d7c4481bf71fca88a0c8f07f25a88d0`
- Four deterministic training seeds: `788101` through `788104`
- 2,048 complete official-engine games per seed, 8,192 games total
- Always second seat, balanced equally between the frozen
  `AshMarnieGeneralV110_extracted` and
  `HelbrosMarnieGeneralRecoveryV4_extracted` opponents
- Public/own-visible information only
- Final iteration only; no intermediate-checkpoint selection
- Six fixed development arms: unchanged control, four seed-final models, and
  the equal-weight parameter mean
- Runtime parity smoke across all arms: 726 matched action decisions and zero
  mismatches
- Locked development panel: 480 games per arm, 2,880 games total
- Bootstrap replicates: 20,000; frozen seed: `789001`

## Development results

| Arm | Win rate | Difference vs control | 95% bootstrap interval | Worst opponent difference | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| control | 43.12% | 0.00 pp | [0.00, 0.00] pp | 0.00 pp | reference |
| seed 788101 | 40.21% | -2.92 pp | [-9.17, 3.33] pp | -5.42 pp | fail |
| seed 788102 | 41.04% | -2.08 pp | [-8.33, 4.17] pp | -4.58 pp | fail |
| seed 788103 | 41.88% | -1.25 pp | [-7.50, 5.00] pp | -2.08 pp | fail |
| seed 788104 | 44.17% | +1.04 pp | [-5.21, 7.29] pp | +0.42 pp | fail |
| mean of all four | 42.71% | -0.42 pp | [-6.67, 5.83] pp | -1.25 pp | fail |

The locked advancement requirements were at least +3.00 percentage points,
a strictly positive 95% bootstrap lower bound, no opponent difference below
-2.00 points, and strict correctness. Every arm passed correctness; no learned
arm passed the strength gates.

## Interpretation

The run was mechanically sound but did not learn a reliable advantage. The
best seed moved both opponent cells slightly in the right direction, which
argues against a total implementation failure, but the effect was small and
statistically unresolved. Three individual seeds regressed, and the parameter
mean did not recover a gain.

The most plausible limitation is the training signal: sparse terminal
win/loss PPO, even with 8,192 games, supplied weak credit assignment for the
long multi-step choices that determine setup preservation and the second ready
attacker. Restricting updates to the final relational block, action-logit head,
and public value head reduced catastrophic drift but also limited strategic
representation change. Repeating this same protocol at modestly larger scale
is therefore not justified.

## Required pivot

N6 should change the learning target, not merely the seed or sample count. The
next campaign should emphasize official-engine branch-conditioned action
values or another dense, decision-level target; train a ranking/advantage
objective that directly distinguishes legal choices; preserve a frozen safe
baseline; and preregister fresh development and sealed-validation boundaries.
Any opponent-exploitation overlay must remain constrained by that safe
baseline. The official-engine adapter, legality path, replay capture,
telemetry, packaging, and benchmark harness remain reusable.

## Durable artifacts

- `N5_PREREGISTRATION_MANIFEST.json`
- `artifacts/team_rocket_recovery/rocket_blueprint_n5/training/`
- `artifacts/team_rocket_recovery/rocket_blueprint_n5/development_arms/arms.json`
- `artifacts/team_rocket_recovery/rocket_blueprint_n5/development_panel/analysis.json`

