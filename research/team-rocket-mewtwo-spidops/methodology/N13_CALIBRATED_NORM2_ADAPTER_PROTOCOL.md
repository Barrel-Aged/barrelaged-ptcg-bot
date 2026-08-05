# N13 Calibrated Norm2 Context-Adapter Protocol

Status: **preregistered before N13 optimization or match evaluation**  
Protocol: `rocket-n13-calibrated-norm2-adapter/v1`  
Preregistered: 2026-08-01

## Single calibrated hypothesis

N12 v3 established a lossless 1,153-parameter context-adapter boundary but
failed to create a candidate: after 400 steps it still fit 0/48 strict
counterfactual corrections. Its correction loss moved only 0.78%, broad KL
remained near zero, and relative parameter displacement was `1.11e-7`.

N13 tests one explicit diagnosis: N12's fixed `5e-7` learning rate was about
two orders of magnitude too small for a behavior-changing intervention. N13 is
not a learning-rate sweep. It runs one path at `5e-5`; no alternative rate,
gradient cap, step count, subset, coefficient, or post-hoc checkpoint is
eligible. Behavior-indexed thresholds and the pre-existing preservation gates
limit the intervention if that calibration is too strong.

## Frozen substrate and evidence

- Exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Live/control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- N12 transfer audit SHA-256:
  `9ef9429934d7d94621406181f61c2a97d57b3409296235373f027efc9e8ed313`.
- N12 v3 training report SHA-256:
  `4a32577c7dbfaec481bf3993de3d89985f0043a7c3483baca6249fb52a5b4947`.
- N9 exact-agent parity SHA-256:
  `ebe70ea381a2f8eef38434ea8ca8ee2f08c867e03a497518520c17188758935e`.
- N9 replay-tree SHA-256:
  `f430676aa3c7786e969d35f0725c70015742adcef5d4e4d4408293e0655cd7ae`.

N13 uses the same 48 strict N6R matched-rollout corrections, the same 591 N6R
preservation roots, and the same 6,753 exact N9 preservation actions. It may
not read the invalid/opaque N12 v2 partial. N9 match outcomes are not training
labels or checkpoint selectors.

## Information and engine boundary

Only exact actor-visible tensors enter training: normal public/own-visible
state plus the actor's persistent public-belief summaries. No realized hidden
hand, prize allocation, visualizer state, or sampled particle identity is
available. Counterfactual targets originate from genuine terminal official
`cg` engine descendants under sampled public-consistent allocations, but the
sampler, estimator, model, and generalization remain research abstractions.

## Model and cache

The post-attention cache and execution are unchanged from the exact N12 v3
boundary. The cached input to `encoder.layers.2.norm2` must reproduce all 7,392
reference decisions with zero top-action mismatches and maximum finite
relative-logit error no greater than `2e-5` before optimization.

Only four tensors, 1,153 parameters total, may change:

- `encoder.layers.2.norm2.weight` and `.bias`;
- `logit_head.weight` and `.bias`.

All other weights, features, belief tracking, legality guards, autoregressive
selection, deck, and runtime assets remain frozen.

## Fixed objective and optimizer

Every step uses all 48 corrections, a deterministic 128-root N6R preservation
batch, and a deterministic 128-action N9 preservation batch. Four
`(opponent, turn)` correction cells have equal mass. Logical anchor means are
accumulated exactly through 32-example microbatches.

- correction max-margin target `0.10`, temperature `0.25`, coefficient `1.0`;
- N6R and N9 forward-KL coefficients `1.0` each;
- N6R and N9 top-action retention coefficients `0.25` each, cap `0.25`;
- relative trainable displacement coefficient `0.01`;
- AdamW learning rate **`5e-5`**, zero optimizer weight decay;
- gradient cap `0.25`, seed `812101`, four CPU threads, exactly 400 steps.

## Behavior checkpoints and offline admission

Save only the first step reaching each teacher-best count: fit12, fit24, fit36,
and fit48. A missing threshold creates no arm; duplicate models retain only the
weaker threshold name. All 400 steps run regardless.

After fresh reload, an arm is admitted only if it retains its named fit count,
changes only the four permitted tensors, retains at least 98.0% top-1 with KL
at most 0.05 on the 591 N6R anchors, retains at least 99.0% overall and 98.5%
per opponent with KL at most 0.02 on all 6,753 N9 actions, and later proves
zero packaged-runtime mismatch/fallback with exact assets. Offline metrics have
no match-strength authority.

## Locked match geometry and confirmation boundary

Control plus every distinct admitted arm enters the same interleaved strict
official-engine panel: 240 second-seat games per opponent, V110 seeds
`813101`/`813102`, Recovery seeds `813201`/`813202`, 120 games per shard.
Outcomes remain unopened until all required shards seal.

Advancement requires at least +2.0 percentage points equally stratified,
strictly positive estimates against both opponents, and a stratified 95% lower
bound above -0.5 points. At most one arm advances by aggregate improvement,
then worst-opponent improvement, then weaker fit threshold.

Any winner still requires at least 3,200 fresh confirmation games, broader
archetype and seat regression, exact package audit, live submission, and 300
rated games. The live bot remains unchanged until every later gate passes.

