# N1 Neural Plan Validation Report

Status: failed; no test, packaging, or arena submission authorized  
Protocol: `rocket-neural-plan-residual/v1`  
Candidate: `seed_781104`  
Analysis SHA-256: `30fc7363f566d42c05ebd4ac9f71baace435f6d8ab209e97fefbafb28e06b3cf`

## Result

The locked validation collected 2,000 fresh complete official-engine games: 500 candidate and 500 control games against each of `AshMarnieGeneralV110` and `HelbrosMarnieGeneralRecoveryV4`. Team Rocket genuinely started second in every game.

| Arm | Games | Wins | Equal-opponent win rate |
|---|---:|---:|---:|
| `seed_781104` | 1,000 | 393 | 39.3% |
| `zero_control` | 1,000 | 423 | 42.3% |

Candidate-minus-control difference: **-3.0 percentage points**.  
Opponent-stratified 95% bootstrap interval (50,000 replicates, seed 784901): **[-7.3, +1.3] percentage points**.

Opponent breakdown:

- `AshMarnieGeneralV110`: 41.2% candidate versus 43.0% control, or -1.8 points.
- `HelbrosMarnieGeneralRecoveryV4`: 37.4% candidate versus 41.6% control, or -4.2 points.

Correctness remained clean: all 2,000 games completed in the official engine, all started second, and exception fallbacks were zero. The candidate intervened in 95.6% of its games.

## Gate audit

- At least +3 pp point improvement: **fail** (-3.0 pp).
- Bootstrap lower bound greater than zero: **fail** (-7.3 pp).
- No opponent regression below -2 pp: **fail** (RecoveryV4 -4.2 pp).
- Official execution and correctness: **pass**.

N1 therefore stops without opening a test set, packaging a submission, or changing the live arena bot.

## Interpretation

The 240-game development gain (+4.58 pp) did not reproduce. N1 used stochastic temperature-1.10 PPO rollouts but deployed a greedy policy; its terminal-reward gradient learned a broad residual that altered almost every game. That combination created a high-variance train/deploy mismatch and a model-selection signal that was too noisy at development scale. The failure is evidence against this N1 objective and intervention regime, not evidence that persistent learned planning is impossible.

The next experiment must be separately preregistered. It should preserve the exact official-engine, legality, deck, telemetry, and validation boundaries while matching the training behavior distribution to greedy deployment, reducing intervention scope, and using substantially stronger variance control.
