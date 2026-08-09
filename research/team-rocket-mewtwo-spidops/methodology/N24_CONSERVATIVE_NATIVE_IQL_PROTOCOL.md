# N24 Conservative Team-Rocket-Native IQL Protocol

Status: **preregistered after train-only feasibility and before N24 held-out evaluation**  
Date: 2026-08-01 (America/Denver)  
Protocol: `rocket-native-conservative-iql/v1`

## Question

Can a Team-Rocket-native action-value learner recover useful temporal credit
from exact live-policy trajectories without repeating the rejected cross-deck
imitation, game-level AWR, localized residual, sampled-branch, or hand-written
rule families?

N24 is a conservative offline-policy-improvement experiment. It is not a
strength claim and has no authority to alter the live arena entry. The live
bot remains `HelbrosRocketSpidopsScaleBCV2` unless every offline, runtime, and
fresh official-engine gate below passes.

## Frozen substrate

- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- qualifying Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- N9 exact-agent parity report SHA-256:
  `ebe70ea381a2f8eef38434ea8ca8ee2f08c867e03a497518520c17188758935e`;
- N9 replay-tree SHA-256:
  `f430676aa3c7786e969d35f0725c70015742adcef5d4e4d4408293e0655cd7ae`;
- frozen N24 train-token cache SHA-256:
  `e6e83143773462e52a0142e718a65c75ec93cc4abe0461f9437dcc42a332c3b0`.

The cache contains 960 decisive strict official-engine games, split evenly
between `AshMarnieGeneralV110_extracted` and
`HelbrosMarnieGeneralRecoveryV4_extracted`, and 55,666 nontrivial
autoregressive choices. Every cached reference argmax is byte-provenance
consistent with the archived action. The frozen live v10 trunk produced all
384-dimensional public-state and option tokens. Private opponent cards,
opponent identity, seed, replay label, and terminal outcome are not runtime
inputs.

The N9 validation and test games were seen by earlier N9 analyses. They are
therefore not claimed to be globally untouched. They remain game-disjoint and
unseen by the N24 optimizer and threshold selector. Fresh official-engine
games, not these reused held-out splits, retain final development authority.

## New mechanism

N24 freezes the full live transformer and learns three separate heads:

1. two independently initialized, deterministic 80%-bootstrap state-action Q
   heads;
2. one upper-expectile public-state V head;
3. one train-only, opponent-blind Platt calibrator for first-choice outcome
   probabilities.

Each Q head is a bounded low-rank interaction between the frozen global token
and each frozen option token, with state bias, option bias, and a learned
reference-logit scale. The supervised behavior-return target is
`outcome * 0.997 ** remaining_choices`. Q loss is smooth L1. The action-value
heads also receive a CQL log-sum-exp penalty over all currently legal options,
which discourages unsupported alternatives. V uses the 0.70 expectile of the
minimum logged-action twin-Q prediction.

This differs from N9 AWR: it does not assign one outcome advantage to every
choice in a game and then imitate it. It learns temporally discounted
state-action returns, an explicit state baseline, action sensitivity, twin
uncertainty, and conservative off-support suppression.

## Frozen optimizer

- seed: `824001`;
- epochs: 12, with no checkpoint selection;
- batch size: 256;
- AdamW learning rate: `3e-4`;
- weight decay: `1e-4`;
- Q bootstrap membership: `((index * 37 + head * 17 + 11) % 5) != 0`;
- discount: `0.997` per supported autoregressive choice;
- expectile: `0.70`;
- CQL coefficient: `0.05`;
- CQL temperature: `0.25`;
- CPU threads: 1;
- equal total training mass per game;
- no validation-based epoch or hyperparameter selection.

## Frozen overlay rule

For a live decision, calculate reference probabilities after the exact deployed
legality and attack-END guards. An alternative is supported only when its
reference probability is at least 2%. Let `L(a) = min(Q1(a), Q2(a))`.

An action can replace the live reference argmax only when:

1. the alternative is supported and legal;
2. it differs from the live argmax;
3. both Q heads individually prefer it to the live argmax; and
4. `L(alternative) - L(live)` meets the frozen margin.

The margin is selected once from train only: the maximum of `0.05` and the
278th-largest eligible train margin, targeting 0.5% of all 55,666 choices. The
train-only selected margin must be serialized in the sole terminal checkpoint;
held-out evidence cannot change it.

## Held-out order and admission gates

Validation contains 120 games and 6,709 choices. Test contains 119 decisive
games and 6,753 choices. Generate their frozen-token caches without reporting
outcome metrics. Open validation first. Do not evaluate test unless every
validation gate passes.

For validation, and then independently for test, require all of:

1. exact reference-action parity: zero mismatches;
2. overall game-level first-choice ROC AUC at least 0.56;
3. worst-opponent first-choice ROC AUC at least 0.53;
4. globally calibrated Brier score better than the frozen train-rate constant;
5. the same opponent-blind calibrator is no worse than each opponent's frozen
   train-rate constant within that opponent;
6. equal-game-weighted logged-action target MAE at most 98% of the within-state
   mean-option-Q target MAE;
7. replacement rate from 0.2% through 1.5%, inclusive;
8. at least two replacements;
9. every replacement has reference support of at least 2%.

Any failure rejects N24 without reinterpretation or threshold adjustment.

## Runtime and official-engine boundary

Passing offline gates is not official candidate execution. If and only if both
held-out splits pass:

1. build a candidate bundle that computes the exact frozen tokens once and
   applies the frozen Q overlay after ordinary deployed legality handling;
2. require deterministic replay parity, zero fallbacks, legal actions, and
   exact agreement with the offline evaluator on every held-out decision;
3. freeze a separate official-engine development-panel lock before games;
4. compare live control and N24 on shared fresh seeds against both frozen
   opponents, initially with the Team Rocket bot in the trained second seat;
5. require positive aggregate improvement, no opponent regression, and the
   separately frozen paired-confidence gates before any broader both-seat
   confirmation.

Only the existing arena adapter, legality handling, telemetry, replay capture,
packaging, and benchmark harness may be reused. Cached observations and model
predictions are abstractions derived from official-engine replays; they are not
new official-engine matches. No Kaggle submission, live replacement, rating,
rank, or performance claim follows from N24 offline admission.

