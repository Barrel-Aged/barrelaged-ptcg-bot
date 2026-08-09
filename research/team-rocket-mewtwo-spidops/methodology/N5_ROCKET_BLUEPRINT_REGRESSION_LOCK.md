# N5 Rocket Blueprint Broader Regression Lock

Status: preregistered before N5 training completed and before any development result

This gate runs only after the locked 3,000-game validation passes. It checks that second-seat Marnie specialization did not destroy the live bot's existing first-seat or non-Marnie strength.

## Panel

The frozen validation candidate and control play 1,200 fresh official-engine games across six equal cells:

- `AshMarnieGeneralV110`, Rocket seat 0, seed 791101;
- `AshMarnieGeneralV110`, Rocket seat 1, seed 791102;
- `HelbrosMarnieGeneralRecoveryV4`, Rocket seat 0, seed 791201;
- `HelbrosMarnieGeneralRecoveryV4`, Rocket seat 1, seed 791202;
- `SparkRocketSpidopsV1`, Rocket seat 0, seed 791301; and
- `SparkRocketSpidopsV1`, Rocket seat 1, seed 791302.

Each cell has 100 games per arm, or 200 match games. Arms are interleaved. Execution uses the official engine, public/own-visible policy features, the greedy packaged-policy path, the attack-safe END guard, and strict no-fallback evaluation.

## Passing rule

Packaging is allowed only if:

- the prior locked N5 validation passed;
- the equal-cell mean candidate-minus-control difference is nonnegative;
- the three-cell first-seat mean difference is at least -2 percentage points;
- no individual opponent/seat cell regresses by more than 10 percentage points; and
- all 1,200 games complete correctly with zero fallbacks.

This regression gate is a safety check, not a fresh optimization surface. It cannot select a different seed, soup, checkpoint, or hyperparameter.
