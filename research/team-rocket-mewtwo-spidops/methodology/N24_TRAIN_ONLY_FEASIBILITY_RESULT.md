# N24 Train-Only Feasibility Result

Status: **feasible direction; no held-out evidence opened and no deployment authority**  
Date: 2026-08-01 (America/Denver)

## Outcome

The new Team-Rocket-native conservative action-value mechanism showed enough
train-only signal to preregister one held-out experiment. This is not a
strength result.

- exact live-action parity: 55,666/55,666 choices;
- twin Q smooth-L1 loss, epoch 1 to epoch 12:
  - Q1: 0.3239 to 0.2151;
  - Q2: 0.3218 to 0.2131;
- overall first-choice train AUC: 0.6669;
- V110 first-choice train AUC: 0.6890;
- RecoveryV4 first-choice train AUC: 0.6451;
- equal-game-weighted logged-action versus mean-option target-MAE ratio:
  0.8970;
- train-only frozen margin: 0.5132653117;
- resulting replacements: 278/55,666 (0.4994%);
- minimum changed-action reference support: 2.0501%.

The raw bounded Q output was rank-informative but not probability-calibrated.
An opponent-blind train-only Platt calibration (`scale=2.0407754098`,
`bias=0.7523584194`) reduced overall train Brier from 0.2593 to 0.2247 versus
the 0.2414 constant baseline. It also beat the frozen train-rate constant in
each opponent stratum. Calibration is now part of the frozen protocol and does
not affect action rankings or the overlay threshold.

## Interpretation

This clears only the feasibility question: the frozen live representation can
separate terminal behavior returns and contains action-specific information
that the within-state mean-option baseline lacks. It does not establish that an
unobserved alternative is causally better. Twin agreement, CQL suppression,
reference support, held-out generalization, runtime parity, and fresh paired
official-engine games are all still mandatory.

The exploratory checkpoint remains under `work/n24_native_iql/feasibility_v1`
and cannot be packaged or submitted. The live bot is unchanged.

