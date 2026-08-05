# N15 Prospective Terminal Consolidation Protocol

Status: **preregistered before N15 implementation, replay, terminal model
construction, or match evaluation**  
Protocol: `rocket-n15-prospective-terminal-consolidation/v1`  
Preregistered: 2026-08-01

## Hypothesis and evidentiary status

N14's first-hit checkpoints established that the rank-16 contextual residual can
fit all 48 strict counterfactual corrections, but those transient policies failed
the locked top-action preservation gates. The already-fixed N14 optimizer then
continued to step 2,000. Its recorded sampled preservation KL fell substantially
while correction fit stayed 48/48, but N14 did not save or evaluate a terminal
model.

N15 prospectively tests exactly one candidate: the step-2,000 terminal state of
an exact from-step-1 replay of the frozen N14 trajectory. There is no new
architecture, hyperparameter, seed, loss, threshold, or checkpoint choice. This
is a development experiment motivated by N14's observed trajectory, not an
independent validation and not evidence of strength by itself.

## Frozen inputs

- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- live/base model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- N14 zero-base model SHA-256:
  `0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb`;
- N14 residual runtime SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`;
- N14 zero-parity report SHA-256:
  `c8638bd6d5a879d32eb88bba30f7ddeb334d222e9b295c64f0ca2619096611d5`;
- N14 protocol SHA-256:
  `2cbf0206af920072705421a349b7d13476766a1eff61dbd0494844bd0ed6821d`;
- N14 training report SHA-256:
  `84ce98182af34919ba753a72be3d71242b31b0d290f16c29c4b77f7729511581`;
- N14 result SHA-256:
  `20efbba7c2775dc5c1d0a767bbb64cd0d950d4cd28e864e70f91e3ba01ea9627`.

The four archived N14 first-hit model hashes are:

- step 62 / `fit12`:
  `e498bac097594901e35ca5f4b57b83ea4b343bb21561abc91cd08fe267ffab0b`;
- step 90 / `fit24`:
  `56a62c22a59f6f9710fee04173172eb9080ea0003b57e71b8d8f2fc5a4ac8c63`;
- step 149 / `fit36`:
  `bea9104a72154c5a609ab0e06d193e3c48468781ca22af8cd219c6eb43571c49`;
- step 286 / `fit48`:
  `0a11204beb06645334647211ca17af978023c864ac74367649e4d7d8451a361f`.

The transfer audit, N6R corpus, N9 parity report, and N9 replay tree remain the
exact inputs transitively frozen by the N14 protocol and trainer. Invalid N12
partials remain excluded.

## Exact replay requirement

N15 constructs the same zero-initialized 6,928-parameter adapter and repeats
the N14 training path verbatim:

- all 48 corrections on every step;
- deterministic 256-example N6R and N9 anchor batches;
- max-margin correction loss target `0.10`, temperature `0.25`, coefficient
  `1.0`;
- N6R and N9 forward-KL coefficients `2.0` each;
- N6R and N9 retention coefficients `1.0` each, target capped at `0.25`;
- relative adapter-displacement coefficient `0.001`;
- AdamW learning rate `0.001`, weight decay `0.0001`;
- gradient cap `1.0`, seed `814001`, four CPU threads, exactly 2,000 steps.

At steps 62, 90, 149, and 286, every model state tensor must be byte-identical
to the correspondingly archived N14 model after reload. The correction-fit
counts must be exactly 12, 24, 37, and 48. Any mismatch invalidates N15, writes
no eligible terminal model, and authorizes no outcome inspection.

Only the step-2,000 terminal state is saved. There are no early-stop,
best-loss, threshold, or alternative terminal arms.

## Offline admission gates

After a fresh reload, the sole terminal candidate is admitted only if all of
the following pass:

- 48/48 strict corrections remain fitted, with every one of the four
  `(opponent, turn)` cells retaining at least one correction;
- only `logit_adapter.*` differs from the zero-residual base;
- N6R non-correction top-1 retention is at least 98.0%, mean forward KL at
  most 0.05;
- N9 top-1 retention is at least 99.0% overall and at least 98.5% in each
  opponent group, mean forward KL at most 0.02;
- full packaged runtime versus contextual-cache execution has zero top-action
  mismatches and finite relative-logit error at most `2e-5` on all 639 N6R
  roots and all 6,753 N9 actions;
- the exact deck and runtime assets remain frozen and strict load/serialization
  checks pass.

Failure of any gate ends N15 without official matches. Offline admission is not
a strength claim.

## Official-engine boundary

The N6R labels originate from genuine official `cg` terminal descendants under
sampled public-consistent hidden allocations. The sampler, counterfactual
estimator, contextual cache, neural residual, and transfer beyond observed roots
are research abstractions. At runtime the actor receives only normal public and
own-visible observations plus its persistent public-belief summary; no realized
opponent hand, prize allocation, sampled-particle identity, or visualizer state
is available.

If the offline gates pass, actual match evidence must come only from strict
official-engine execution. Cached logits and training losses never count as
games or wins.

## Locked development panel if admitted

The frozen live control and sole N15 terminal candidate each receive 240
second-seat games against each opponent, split into two 120-game shards:

- V110 seeds `815101` and `815102`;
- RecoveryV4 seeds `815201` and `815202`.

All eight required shards must finish and seal before outcome analysis.
Correctness requires exact bot/deck/model hashes, zero illegal/fallback events,
complete seed/game coverage, and official-engine replay capture.

Advancement requires all of:

- equally stratified candidate-minus-control win-rate improvement at least
  +2.0 percentage points;
- strictly positive improvement against each opponent;
- stratified paired-bootstrap 95% lower bound greater than -0.5 percentage
  points.

A development pass still requires a fresh, uninterrupted confirmation of at
least 3,200 games, broader opponent/seat regression, package audit, and then
live-arena evidence. The current live bot remains unchanged unless every later
gate passes.
