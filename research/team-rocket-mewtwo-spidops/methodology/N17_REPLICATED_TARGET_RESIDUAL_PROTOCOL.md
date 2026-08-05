# N17 Replicated-Target Residual Protocol

Status: **preregistered before N17 model construction, optimization, or match
evaluation**  
Protocol: `rocket-n17-replicated-target-residual/v1`  
Preregistered: 2026-08-01

## Hypothesis

N14-N16 trained a localized residual on only 48 roots whose clustered 95%
lower bound was strictly positive. That family was directionally useful but too
small: N16 was positive against both development opponents yet gained only 1.67
points equally stratified.

The original N6R official-descendant corpus contains a second, independent
reliability signal that was not used by those residuals. Each root was evaluated
under two disjoint six-allocation halves. At 160 roots, both halves selected the
same non-baseline full-sample best action. The frozen union of these replicated
roots and the strict roots contains 170 targets, balanced across opponents and
turns. N17 tests whether a higher-capacity localized residual can learn that
broader signal while preserving the live policy elsewhere.

This target expansion is defined entirely from old counterfactual evidence. No
N15/N16 match row, win, model comparison, or outcome enters the target audit or
trainer.

## Frozen substrate and target audit

- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- live/base model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- residual-capable runtime SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`;
- N17 target audit SHA-256:
  `c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c`;
- target-audit script SHA-256:
  `c4c9f219a894a8847a8a308f10b85623f0cb8efe9645f8b3bfb7b4e289bfa003`;
- N6R train analysis SHA-256:
  `a9b0dc1fd9414b63e327b35616455f7ff699dc69ea1833956add461aee8c3049`;
- N6R selection analysis SHA-256:
  `b13e8d804b466230d7019f86d018e9db1ef9d0de521ce71565804a0988c9acbd`;
- N12 exact transfer audit SHA-256:
  `9ef9429934d7d94621406181f61c2a97d57b3409296235373f027efc9e8ed313`.

The target criterion is exactly:

```text
strict_admissible_change
OR
(teacher_best != baseline
 AND half_best[0] == half_best[1] == teacher_best)
```

The immutable audit proves:

- 639 unique source roots and 170 union targets;
- 48 strict targets, 160 replicated-half targets, overlap 38;
- 469 non-target N6R anchors;
- targets by opponent: 83 V110, 87 RecoveryV4;
- targets by turn: 91 turn 2, 79 turn 4;
- four cells: V110/t2 45, V110/t4 38, Recovery/t2 46, Recovery/t4 41;
- every target has all 24 matched pairs and all 12 allocation clusters.

The prior N9 exact replay support remains the same 6,753 actions and hashes
transitively frozen by N14-N16.

## Fixed zero-residual architecture

Build a fresh rank-32 residual on the frozen 384-dimensional contextual option
embedding:

```text
residual(option) = Linear(32 -> 1, no bias)(
    GELU(Linear(384 -> 32, no bias)(LayerNorm(384)(option)))
)
candidate_logit = frozen_live_logit + residual(option)
```

The final projection is exact zero. LayerNorm and the down projection use
PyTorch initialization under seed `817001`. The adapter has exactly 13,088
parameters. Construction must preserve every original model tensor and have
zero relative-logit and top-action error on all 639 N6R roots and 6,753 N9
actions.

Only `logit_adapter.*` may train. Runtime defaults for legacy models remain
unchanged.

## Fixed target weighting and optimization

The strict stratum contains all 48 strict targets. The expanded-only stratum
contains the other 122 union targets. Each stratum receives total loss mass
one, divided equally across its four `(opponent, turn)` cells and then equally
within a cell.

- strict max-margin target `0.10`, temperature `0.25`, coefficient `1.0`;
- expanded-only max-margin target `0.05`, temperature `0.25`, coefficient
  `1.0`;
- deterministic N6R anchor batch 384, forward-KL coefficient `3.0`, retention
  coefficient `1.5`;
- deterministic N9 anchor batch 384, forward-KL coefficient `3.0`, retention
  coefficient `1.5`;
- retention target capped at the reference margin or `0.25`;
- relative adapter-displacement coefficient `0.001`;
- AdamW learning rate `0.001`, weight decay `0.0001`, gradient cap `1.0`;
- training seed `817002`, four CPU threads, exactly 4,000 steps.

Training always completes. Save only the step-4,000 terminal model. No
first-hit, best-loss, early-stop, alternate rank, alternate seed, or alternate
hyperparameter model exists.

## Offline admission

After fresh reload, the sole terminal model is admitted only if:

- all 48 strict targets are top-1;
- at least 110/122 expanded-only targets and at least 85% of expanded-only
  targets within each of the four cells are top-1;
- only `logit_adapter.*` differs from the exact zero base;
- N6R non-target top-1 retention is at least 98.0%, mean forward KL at most
  0.05;
- N9 top-1 retention is at least 99.0% overall and 98.5% per opponent, mean
  forward KL at most 0.02;
- cached/full-runtime execution has zero top-action mismatch and finite
  relative-logit error at most `2e-5` on all 639 N6R and 6,753 N9 actions;
- serialization is tensor-exact and exact deck/runtime assets remain frozen.

Failure ends N17 before match play. Offline admission is not a strength claim.

## Information and official-engine boundary

The labels derive from genuine official `cg` terminal descendants under sampled
public-consistent hidden allocations. The hidden-allocation prior,
counterfactual estimator, replicated-half criterion, contextual cache, and
neural transfer are research abstractions. Runtime receives only normal public
and own-visible observations plus the persistent public-belief summary. No
realized opponent hand, prize allocation, particle identity, or visualizer state
is available.

## Fresh development panel if admitted

The live control and sole N17 candidate each receive 240 second-seat games per
opponent in two 120-game shards:

- V110 seeds `817101` and `817102`;
- RecoveryV4 seeds `817201` and `817202`.

After all shards seal, analyze 20,000 stratified bootstrap replicates under seed
`817001`. Advancement requires strict correctness, at least +2.0 percentage
points equally stratified, strictly positive improvement against both
opponents, and a 95% lower bound greater than -0.5 points.

A pass permits only at least 3,200 fresh confirmation games plus broad both-seat
regression and package audit. The live bot remains unchanged until every later
gate passes.
