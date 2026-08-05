# Target-80 residual advantage V2 protocol

Frozen: 2026-08-05, before V2 fitting or V2 development evaluation  
Status: **prospective development protocol; no submission authority**

## Why V2 exists

Randomized Advantage V1 repaired the action-support deficit but its joint Q
heads still derived most discrimination from public-state strength. On V1
validation, Q AUC on exploratory actions remained below state-only AUC for all
three configurations. Full inverse-propensity weighting also produced an
overall effective-sample ratio of 0.242 and a minimum cell ratio of 0.210.

V2 isolates action signal instead of loosening those failed gates.

## Frozen implementation and data roles

- V2 learner SHA-256:
  `d66f04256c2e6246902b6bb39d7461d184058536176e8148f091509455b66693`
- train cache SHA-256:
  `b2bc46ce2d46bf5e9733051ca09c1e15c35fca7a24efeb0cb5801b168d1d038a`
- V2 development cache SHA-256:
  `886973ccb85bb98ffd971bfb487240b9a862f3df79b6f84aa65d943be0db34a7`
- source corpus SHA-256:
  `3dde57e3420070bf43f31ffd9fa38ac6a9015c6fa6cc3878a63dff776eb69602`

The 720-game train cache is unchanged. The 240-game cache originally named
`test_v2.pt` is demoted to **V2 development validation** because its aggregate
outcome count was exposed during cache construction. It has never been used to
fit any state or action parameter and no V1 checkpoint was evaluated on it.
It cannot authorize V2 confirmation or submission.

## Cross-fitted state baseline

Train a state-only public outcome network first:

- five game-level folds assigned by `(game_id * 37 + 13) mod 5`;
- width 128, eight epochs per fold, AdamW `3e-4`, weight decay `1e-4`;
- equal total base mass per game;
- seeds 855100 through 855104 for folds and 855200 for the final all-train
  deployment state model;
- target `+1` win, `0` draw, `-1` loss;
- out-of-fold predictions for every train choice;
- train-only Platt calibration fitted to out-of-fold predictions.

The action heads never receive an in-fold fitted state residual. The final
state model is used only on held-out development/confirmation states.

## Residual action heads

For each eligible randomized choice, freeze the state baseline and train twin,
independently bootstrapped state-option heads on:

`terminal outcome - cross-fitted state prediction`.

Each head is centered to zero under the exact logged behavior distribution.
Training retains equal base mass per game and applies a prospectively frozen
square-root inverse-propensity multiplier `1/sqrt(p)`, capped at 4. This is a
deliberate variance/bias tradeoff, not a claim of fully unbiased inverse-
propensity estimation. Before outcomes were used, the resulting support-only
ESS ratios were checked mechanically: minimum 0.406 on train and 0.295 on the
V2 development cache, compared with V1's 0.210 minimum.

All action heads use width 96, ten epochs, AdamW `3e-4`, weight decay `1e-4`,
and deterministic 80% bootstrap masks.

| Configuration | Residual regularization | Seed |
|---|---:|---:|
| `low_reg` | 0.01 | 855201 |
| `balanced` | 0.03 | 855202 |
| `high_reg` | 0.08 | 855203 |

No additional V2 configuration may be introduced after development validation
is opened.

## Development selection gates

The state model and all three frozen action configurations are evaluated on the
same 240-game V2 development cache. A configuration qualifies only if:

1. the overall and every opponent/seat effective-sample ratio are at least
   0.25;
2. on realized exploratory actions, Q AUC exceeds state-only AUC by at least
   0.002;
3. on realized exploratory actions, Q Brier is no worse than state-only Brier;
4. on realized exploratory actions, predicted-advantage residual MAE is lower
   than the zero-advantage residual MAE;
5. first-eligible-per-game Q AUC is no worse than state-only AUC;
6. a frozen twin-agreement margin in
   `0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30`
   changes between 0.5% and 5% of supported choices; and
7. all proposed changes remain in logged support and both heads select the same
   alternative.

Select the qualifying configuration with the largest exploratory Q-minus-state
AUC gain; break ties by lower Q Brier, lower residual MAE, then stronger
regularization. Select the largest frozen margin retaining at least a 0.5% and
at most a 5% change rate. Freeze configuration, margin, state hash, action-head
hash, and runtime semantics before collecting confirmation.

If no configuration qualifies, V2 closes without test-driven tuning.

## Fresh confirmation requirement

Because the prior test aggregate was exposed, successful V2 development must
be followed by a newly collected 240-game official-engine confirmation set:
40 games for each Marnie/Starmie/Alakazam and Rocket-seat cell, using the exact
V1 behavior and new locked seeds. No confirmation outcome may be inspected
until all six shards are terminal and independently audited. The selected V2
model receives one evaluation only.

Confirmation must reproduce every development gate, and checkpoint reload must
reproduce Q values, best supported actions, and frozen-margin changes exactly.
Only then may a fail-closed runtime overlay enter a new preregistered
official-engine matchup panel. Offline evidence alone never authorizes an arena
submission.

