# N19 Retention-Constrained Leader Behavior-Cloning Protocol

Status: **preregistered before training**  
Protocol: `rocket-n19-retention-constrained-leader-bc/v1`

## Rationale

N18 decisively rejected raw current-leader parameter and recurrent-residual
transfer to the exact Team Rocket deck.  The direct leader base was −16.53
percentage points and the leader base plus residual was −11.25 points versus
the unchanged live control.  N19 does not reuse or soften those arms.

The historical mechanism that produced the strongest Team Rocket submission
was behavior cloning of winning public leader actions while starting from a
general relational policy.  N19 tests one narrow version of that mechanism:
start from the current live Rocket model, learn a small amount from fresh
winning current-leader actions, and simultaneously constrain the model to
retain its current behavior on a disjoint, balanced set of exact-Rocket
official-engine observations.

This is a supervised policy experiment.  It is not a hidden-hand belief model,
tree search, counterfactual solver, or claim that leader actions from another
deck are optimal for Team Rocket.

## Frozen evidence

- N18 result SHA-256:
  `77f02de76e3b7e7f840800cbae4a6580f7af5ffe1dc0ff43fab8452cb855aa15`;
- N19 corpus auditor SHA-256:
  `86f6e12bb96c16ebfa0980e27974bff881537cb8e750e9c5adc6a688bfda58c1`;
- N19 corpus audit SHA-256:
  `d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed`;
- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- leader base model SHA-256:
  `6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

## Frozen corpus

The source is the 984-game deduplicated N18 public-arena corpus.  Recorded
actions are paired with their same-row official-engine observations and have
already passed exact legality checks.

The teacher set contains exactly 120 winning games by the one shared
rank-one/rank-two strategy, excluding every game containing the live Rocket
bot.  It is balanced at 60 games per teacher seat.  Within each seat, the first
60 sorted payload hashes are split 48/6/6 into train/validation/test, yielding:

- 96 train games and 9,483 autoregressive choice targets;
- 12 validation games and 1,194 targets;
- 12 test games and 1,138 targets.

The retention set is separate and contains exactly 120 direct games involving
the live Rocket bot.  It is balanced at 30 games in each live-seat-by-outcome
stratum.  Each stratum is split 24/3/3, yielding:

- 96 train games and 6,240 targets;
- 12 validation games and 713 targets;
- 12 test games and 801 targets.

Teacher and retention payload hashes have zero overlap.  Optional empty
selections, early stops, and trivial one-option decisions are excluded because
the V10 action head cannot represent them.  Fully supported multi-select
actions are unrolled autoregressively.  The deployed attack guard is applied
to feature masks.

Baseline validation top-1 is 76.30% for live on teacher targets and 96.63% for
live on recorded retention targets.  The raw leader base reaches 87.77% on the
teacher targets but only 64.66% on retention targets, consistent with N18's
large exact-Rocket regression.  N19 must move toward teacher behavior without
approaching that raw transfer.

## Sole training run

There is exactly one training run and one eligible terminal model.  No grid,
checkpoint selection, seed selection, loss-weight selection, or post-outcome
rescaling is allowed.

- initialization: exact live model;
- trainable tensors: the final transformer block plus `logit_head`, exactly 14
  tensors and 1,184,257 parameters;
- frozen tensors: all other 45 model tensors;
- epochs: exactly 4;
- batch size: 96 teacher examples plus 96 retention examples per optimizer
  step;
- teacher order: deterministic epoch shuffle under seed `819001 + epoch`;
- retention order: deterministic independent epoch shuffle under seed
  `819101 + epoch`, cycled as needed;
- optimizer: AdamW;
- learning rate: `2e-6`;
- weight decay: `1e-4`;
- gradient norm cap: `1.0`;
- teacher label smoothing: `0.01` over legal actions only;
- retention KL coefficient: `4.0`;
- teacher-state trust-region KL coefficient: `0.25`;
- source-parameter mean-square anchor coefficient: `0.01`;
- CPU threads: 6;
- checkpoint rule: terminal epoch 4 only.

For each step, the loss is:

`teacher_cross_entropy + 0.25 * teacher_KL_to_live + 4.0 * retention_KL_to_live + 0.01 * source_parameter_anchor`.

KL uses the frozen live model's legal-action distribution on the exact same
feature batch.  The parameter anchor is the mean squared difference from the
frozen live values over only the 14 trainable tensors.  No reward, game result,
opponent private card, N18 outcome, search value, or hand hypothesis enters the
gradient.

## Terminal offline gates

The epoch-4 model receives offline development authority only if every gate
passes:

1. model configuration, key set, tensor shapes, exact deck, and serialization
   are valid;
2. only the 14 preregistered tensors differ from live;
3. teacher validation top-1 improves by at least +2.0 percentage points and
   teacher validation cross-entropy decreases;
4. teacher test top-1 improves by at least +1.0 percentage point and teacher
   test cross-entropy decreases;
5. base-policy top-1 agreement is at least 99.5% on each fresh retention split;
6. mean KL from the base policy is at most 0.01 on each fresh retention split;
7. recorded-action retention top-1 on validation and test is no more than 0.5
   percentage points below the frozen live baseline;
8. the established N6 exact-Rocket anchors retain at least 99.0% base top-1;
9. the established N9 exact-Rocket trajectory anchors retain at least 99.0%
   overall base top-1 and at least 98.5% in every frozen N9 group;
10. no extraction, feature, legality, finite-value, or serialization error is
    present.

Failure closes N19.  Offline passage grants only authority to freeze one
official-engine development panel comparing this terminal model with the live
control.  It does not grant confirmation, packaging, submission, deployment,
rank, strength, or superhuman claims.

## Future official-engine boundary

If and only if all offline gates pass, a separate durable lock must freeze the
candidate package, runtime parity, opponents, both-seat allocation, seeds,
sample size, analysis, and admission thresholds before games begin.  Fresh
native-engine outcomes—not teacher imitation—determine whether the candidate
has any strategic value for Team Rocket.

The existing live submission remains unchanged throughout N19 unless a later
development panel, independent confirmation, and broad both-seat regression
all pass.
