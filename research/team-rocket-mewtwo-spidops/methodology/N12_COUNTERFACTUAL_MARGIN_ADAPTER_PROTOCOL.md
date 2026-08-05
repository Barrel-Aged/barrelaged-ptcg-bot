# N12 Counterfactual-Margin Context Adapter Protocol

Status: **preregistered before N12 optimization or match evaluation**  
Protocol: `rocket-n12-counterfactual-margin-adapter/v1`  
Preregistered: 2026-08-01

## Purpose

N11 independently disproved the apparent N9 epoch-4 development gain. N12 does
not reuse N9's outcome-weighted action-head objective and does not treat the
N6R sampled-rollout policy as a deployable teacher. It asks a narrower causal
question: can the current live model learn the 48 still-unlearned N6R roots
whose matched official-engine continuations passed the preregistered N6R hard
change rule, while preserving current behavior on a much broader exact replay
support?

The N12 transfer audit is immutable input evidence, not a strength result. It
found that the current live policy exactly reproduces the old baseline action
on all 639 roots, that its action remains in every frozen three-action
shortlist, and that all 48 admissible counterfactual corrections remain new.
The 48 roots contain 28 V110 and 20 Recovery examples, split across turns 2 and
4. Each source estimate came from 24 matched hidden-allocation/coin comparison
sets and genuine official-engine terminal continuations.

## Frozen substrate

- Exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Current live/control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Current live/control directory:
  `artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2`.
- N12 transfer-audit SHA-256:
  `9ef9429934d7d94621406181f61c2a97d57b3409296235373f027efc9e8ed313`.
- N9 replay-tree SHA-256:
  `f430676aa3c7786e969d35f0725c70015742adcef5d4e4d4408293e0655cd7ae`.
- N9 exact-agent parity SHA-256:
  `ebe70ea381a2f8eef38434ea8ca8ee2f08c867e03a497518520c17188758935e`.

The N6R root and manifest hashes must match the hashes already bound by the
transfer audit. The N9 deterministic test split is reused only as a behavior
preservation support: 6,753 exact action examples (3,456 V110 and 3,297
Recovery) from 119 complete decisive games. The corpus's one immutable draw is
excluded by the pre-existing N9 binary split before this support is formed.
Their match outcomes are not inputs to the
N12 objective, checkpoint schedule, admission gates, or match analysis.

## Actor-information and engine boundary

N12 receives only the exact feature tensors that the frozen actor possessed at
the decision. The N6R snapshots contain normal public/own-visible state and the
actor's persistent public-belief features; they contain no realized opponent
hand, visualizer state, or sampled hidden allocation. N9 preservation examples
are reconstructed by the previously audited exact deployed actor path.

The 24-way counterfactual values are genuine descendant executions of the
official `cg` engine from sampled public-consistent information states. The
sampling distribution, hard-change estimator, neural loss, and any later
policy generalization are research abstractions. N12 is not online tree search,
not an official hidden-state oracle, and not a claim that the sampled belief is
the true posterior.

## Fixed learner

The reference model stays in evaluation mode. Only the final relational
Transformer block and `logit_head` may change. Every embedding, projection,
earlier Transformer block, auxiliary/value head, feature extractor, belief
tracker, legality guard, and autoregressive selection path remains byte-frozen.

Every optimization step uses:

1. all 48 hard-admissible corrections;
2. a deterministic 128-example batch cycled through the other 591 N6R roots;
3. a deterministic 128-example batch cycled through the 6,753 N9 test actions.

The correction loss is a temperature-smoothed max-margin loss between the
teacher-best logit and the strongest other deployable action. Its target margin
is `0.10` and softplus temperature is `0.25`. Correction examples are weighted
so each of the four `(opponent, turn)` cells contributes equal total mass.

Both preservation batches use frozen-reference forward KL. They also use a
top-action retention loss whose target is the smaller of the reference top-two
margin and `0.25`. Fixed coefficients are:

- correction margin: `1.0`;
- N6R preservation KL: `1.0`;
- N9 preservation KL: `1.0`;
- N6R retention: `0.25`;
- N9 retention: `0.25`;
- relative trainable-parameter displacement: `0.01`.

Optimization is AdamW with learning rate `5e-7`, zero optimizer weight decay,
gradient-norm cap `0.25`, seed `812101`, one CPU thread, and at most 400 steps.
The Transformer remains deterministic with deployment-time dropout disabled.

## Behavior-indexed checkpoints

Checkpoint strength is fixed by causal-correction attainment, not by match
outcomes or an offline scalar score. Save the first optimization step reaching
each cumulative threshold:

- `fit12`: at least 12 of 48 correction roots choose teacher-best;
- `fit24`: at least 24 of 48;
- `fit36`: at least 36 of 48;
- `fit48`: all 48.

If a threshold is never reached, that arm does not exist. If two thresholds
produce byte-identical models, retain only the weaker named threshold. Training
continues to step 400 even after all checkpoints exist so the execution report
has a single predetermined terminal condition. No later step, learning rate,
loss coefficient, subset, or checkpoint may be selected.

## Offline admission

Each saved checkpoint is evaluated from a fresh load and may enter a match
panel only if all of the following pass:

- only the permitted final block and action-head state keys changed;
- it still meets its named correction-fit threshold;
- top-1 retention is at least 98.0% on the 591 non-correction N6R roots;
- top-1 retention is at least 99.0% over all 6,753 N9 test actions;
- top-1 retention is at least 98.5% within each N9 opponent stratum;
- mean forward KL is at most 0.05 on N6R non-corrections;
- mean forward KL is at most 0.02 on N9 test actions;
- deck and all packaged execution assets remain exact; and
- strict packaged-runtime parity has zero mismatches and zero fallbacks.

Offline admission is a safety filter only. It cannot establish strength.

## Locked development geometry

Control plus every admitted distinct checkpoint enters one interleaved,
official-engine development panel. Rocket is always seat 1 and the official
first player is seat 0. Each arm receives 240 games against each frozen
opponent (480 total), divided into two 120-game shards:

- V110 seeds `813101` and `813102`;
- Recovery V4 seeds `813201` and `813202`.

The arm order, collector/analyzer hashes, exact commands, and complete arm
manifest must be sealed in an execution lock before the first panel game. No
outcome may be inspected until all required shards are complete and sealed.

An arm advances only with strict correctness, 480 games, at least +2.0
percentage points equally stratified improvement over control, a strictly
positive point estimate against both opponents, and an opponent-stratified
95% bootstrap lower bound greater than -0.5 percentage points. At most one arm
advances, breaking a tie by aggregate improvement, then worst-opponent
improvement, then the weaker correction threshold.

## Post-development boundary

Any development winner must pass a fresh independent confirmation of at least
3,200 official-engine games, regression against additional archetypes and both
seats, exact package/deck audit, and live submission monitoring. N12 cannot
replace the arena bot or support a strength claim from offline metrics or one
development panel. The current live bot remains unchanged unless every later
gate passes.
