# N12 Norm2 Context-Adapter Protocol

Status: **preregistered before v3 optimization or match evaluation**  
Protocol: `rocket-n12-norm2-margin-adapter/v3`  
Preregistered: 2026-08-01

## Motivation and clean-result boundary

N12 v1 and v2 attempted to train the complete final 384-wide Transformer
block. Both were computationally invalidated before a stdout loss, fit count,
or checkpoint metric was received. The v1 directory is empty. During v2
termination one partial file appeared, but it was never named, opened, hashed,
loaded, or analyzed; its entire directory is preserved opaque at
`artifacts/team_rocket_recovery/n12_counterfactual_margin_adapter/training_v2_cached_invalid_uninspected`
and is excluded from all future inputs and claims.

V3 changes model capacity for a reason known without model outcomes: full
self-attention backpropagation made a 400-step run operationally infeasible.
It trains only the final block's second LayerNorm and the action head. This is a
1,153-parameter context adapter: the frozen attention stream is already
state- and option-conditioned, LayerNorm can reweight its 384 contextual
channels, and the head converts those channels to action scores. It is more
expressive than N9's 385-parameter head-only update but far narrower than the
failed N6R full-final-block distillation.

## Frozen substrate and data

- Exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Live/control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Transfer-audit SHA-256:
  `9ef9429934d7d94621406181f61c2a97d57b3409296235373f027efc9e8ed313`.
- N9 exact-agent parity SHA-256:
  `ebe70ea381a2f8eef38434ea8ca8ee2f08c867e03a497518520c17188758935e`.
- N9 replay-tree SHA-256:
  `f430676aa3c7786e969d35f0725c70015742adcef5d4e4d4408293e0655cd7ae`.

The correction support remains all 48 N6R roots that passed the original hard
change rule: 24 matched official-engine comparison sets per root, both hidden-
allocation halves nonnegative, and a strictly positive clustered-bootstrap
95% lower bound. The preservation support remains the other 591 exact N6R
roots plus 6,753 exact N9 test-split actions (3,456 V110 and 3,297 Recovery).
Match outcomes are not used by the v3 loss or checkpoint schedule.

## Actor-information and official-engine boundary

The learner sees only exact actor features containing normal public/own-visible
state and persistent public-belief summaries. It never sees the opponent's
realized hand, prize allocation, visualizer state, or sampled particle
identity. N6R counterfactual descendants are genuine official `cg` engine
executions from sampled public-consistent information states. The belief
sampler, hard-change statistic, neural adapter, and generalization outside the
48 roots remain research abstractions rather than official hidden-state truth.

## Frozen post-attention cache and trainable scope

The reference model runs once per example. A pre-hook captures the exact input
to `encoder.layers.2.norm2`, which is the residual stream after the final
block's frozen self-attention sublayer because the architecture uses
`norm_first=True`. V3 then executes:

`x + frozen_feed_forward(trainable_norm2(x)) -> trainable_logit_head`.

The feed-forward block is token-local and dropout is zero, so masked padding
tokens can be removed without changing any real token. Before optimization,
the cached path must have zero top-action mismatches and maximum finite
relative-logit error no greater than `2e-5` on all 7,392 examples.

Only these state keys may change:

- `encoder.layers.2.norm2.weight`;
- `encoder.layers.2.norm2.bias`;
- `logit_head.weight`;
- `logit_head.bias`.

All other model parameters, feature/belief code, legality guards,
autoregressive selection, deck, and packaged assets remain byte-frozen.

## Fixed objective and optimization

Every step uses all 48 corrections, a deterministic 128-example batch cycled
through the 591 N6R preservation roots, and a deterministic 128-example batch
cycled through the 6,753 N9 preservation actions. Correction weight is equal
across the four `(opponent, turn)` cells.

The loss and coefficients are unchanged from the cleanly preregistered N12 v1
scientific design:

- smoothed teacher-best versus strongest-other margin, target `0.10`, softplus
  temperature `0.25`, coefficient `1.0`;
- frozen-reference forward KL coefficient `1.0` on each preservation support;
- top-action retention coefficient `0.25` on each support, targeting the
  smaller of its frozen top-two margin and `0.25`;
- relative trainable-parameter displacement coefficient `0.01`.

Each 128-example logical anchor mean is accumulated through four 32-example
microbatches with exact size weighting before one optimizer step. Optimization
is AdamW, learning rate `5e-7`, zero optimizer weight decay, gradient cap
`0.25`, seed `812101`, four CPU threads, exactly 400 steps.

## Behavior-indexed checkpoints and offline gates

Save only the first steps that reach 12, 24, 36, and 48 teacher-best
corrections. Missing thresholds produce no arm; byte-duplicate thresholds keep
only the weaker name. Training always continues through step 400.

Each saved arm must pass all fixed offline gates:

- named correction threshold remains met after reload;
- changed keys are exactly within the four-key scope above;
- N6R non-correction top-1 retention at least 98.0% and mean KL at most 0.05;
- N9 top-1 retention at least 99.0% overall and 98.5% per opponent, with mean
  KL at most 0.02;
- zero packaged-runtime mismatches/fallbacks and exact deck/assets.

These gates authorize only development testing, not a strength claim.

## Official-engine development and later confirmation

Control plus every distinct admitted arm receives 240 second-seat games per
opponent, using V110 seeds `813101`/`813102` and Recovery seeds
`813201`/`813202`, 120 games per shard. Outcomes stay unopened until every
required shard seals. Advancement requires 480 games, strict correctness,
at least +2.0 percentage points equally stratified improvement, a strictly
positive estimate against both opponents, and a stratified 95% bootstrap lower
bound above -0.5 points. At most one arm advances.

Any development winner still needs at least 3,200 fresh official-engine
confirmation games, broader archetype/seat regression, exact package audit,
live submission, and 300 rated games. The current live bot is unchanged unless
all later gates pass.

