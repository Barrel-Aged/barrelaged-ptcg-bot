# N14 Localized Contextual Residual Protocol

Status: **preregistered before N14 model construction, optimization, or match
evaluation**  
Protocol: `rocket-n14-localized-contextual-residual/v1`  
Preregistered: 2026-08-01

## Hypothesis

N13 moved 11/48 strict corrections but incurred broad preservation drift before
reaching fit12. Its four trainable tensors globally reweighted the same
contextual channels for every action. N14 tests a materially different
hypothesis: a small nonlinear residual scorer can recognize correction-like
state/option embeddings while leaving the frozen live scorer as the baseline.

This is not a search or learning-rate sweep. One architecture, one seed, one
optimizer path, and the existing behavior-indexed checkpoints are eligible.

## Frozen substrate and supports

- Exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Live/base model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Live/base `soldier/net_v10.py` SHA-256:
  `130061f3a064c32b7f1496c2e11f472359686e9185dd42b7bf488b898f5b8762`.
- Transfer-audit SHA-256:
  `9ef9429934d7d94621406181f61c2a97d57b3409296235373f027efc9e8ed313`.
- N13 result report SHA-256:
  `d99d5552a069f8e18b34c85644832b89970250fe80d2bba8dcdcbd8709e0a3ea`.
- N9 parity SHA-256:
  `ebe70ea381a2f8eef38434ea8ca8ee2f08c867e03a497518520c17188758935e`.
- N9 replay-tree SHA-256:
  `f430676aa3c7786e969d35f0725c70015742adcef5d4e4d4408293e0655cd7ae`.

Training uses all 48 hard-admissible N6R corrections. Preservation uses the
other 591 exact N6R roots and all 6,753 exact N9 test actions. Invalid N12
partials and every prior candidate model are excluded. Outcomes do not select
an architecture, step, or checkpoint.

## Information and official-engine boundary

Every input is an exact normal actor feature: public/own-visible state plus the
actor's persistent public-belief summary. No realized opponent hand, prize
allocation, visualizer state, or sampled-particle identity is used. The N6R
targets come from genuine official `cg` terminal descendants under sampled
public-consistent allocations; the sampler, hard-change estimator, residual
model, and transfer beyond observed roots are research abstractions.

## Fixed residual architecture

The complete frozen live network produces each 384-dimensional contextual
option embedding and its baseline logit. N14 adds:

```text
residual(option) = Linear(16 -> 1, no bias)(
    GELU(Linear(384 -> 16, no bias)(LayerNorm(384)(option)))
)
candidate_logit = frozen_live_logit + residual(option)
```

The final 16-to-1 matrix is initialized to exact zeros, so construction must
have byte-equivalent choices and zero relative-logit error on all 7,392 support
examples. LayerNorm and the down projection use PyTorch initialization under
seed `814001`. Rank is fixed at 16 and residual scale at 1.0. The adapter has
exactly 6,928 trainable parameters.

Only `logit_adapter.*` may change. All original model tensors, feature/belief
code, legality guards, autoregressive semantics, and deck assets remain
byte-frozen. The packaged `RelationalNet` extension must default to rank zero,
so legacy model behavior and loading remain unchanged.

## Fixed training path

Every step uses all 48 corrections, a deterministic 256-root N6R preservation
batch, and a deterministic 256-action N9 preservation batch. Correction mass
is equal across the four `(opponent, turn)` cells.

- correction max-margin target `0.10`, temperature `0.25`, coefficient `1.0`;
- N6R forward KL coefficient `2.0` and retention coefficient `1.0`;
- N9 forward KL coefficient `2.0` and retention coefficient `1.0`;
- retention target capped at the reference margin or `0.25`;
- relative adapter displacement coefficient `0.001`;
- AdamW learning rate `0.001`, weight decay `0.0001`;
- gradient cap `1.0`, seed `814001`, four CPU threads, exactly 2,000 steps.

## Checkpoints and offline gates

Save only the first steps reaching fit12, fit24, fit36, and fit48. Missing
thresholds produce no arm; byte-identical checkpoints retain only the weaker
name. Training always completes 2,000 steps.

After fresh reload, an arm is admitted only if:

- it still meets its named fit threshold;
- only `logit_adapter.*` differs from the zero-residual base;
- N6R non-correction top-1 retention is at least 98.0%, mean KL at most 0.05;
- N9 top-1 retention is at least 99.0% overall and 98.5% per opponent, mean KL
  at most 0.02;
- zero-residual and trained full-runtime versus cached-runtime parity have zero
  top-action mismatches and relative-logit error at most `2e-5`;
- strict packaged execution has zero mismatch/fallback and exact deck/assets.

Offline admission is not match-strength evidence.

## Locked official-engine panel

Control plus every distinct admitted arm receives 240 second-seat games per
opponent, in two 120-game shards: V110 seeds `813101`/`813102`, Recovery seeds
`813201`/`813202`. All required shards must seal before outcomes are opened.

Advancement requires strict correctness, at least +2.0 percentage points
equally stratified, a strictly positive point estimate against both opponents,
and a stratified 95% bootstrap lower bound greater than -0.5 points. At most one
arm advances, breaking ties by aggregate improvement, worst-opponent
improvement, then weaker fit threshold.

Any development winner still needs at least 3,200 fresh confirmation games,
broad archetype/seat regression, exact package audit, live submission, and 300
rated games. The current live bot remains unchanged until every gate passes.

