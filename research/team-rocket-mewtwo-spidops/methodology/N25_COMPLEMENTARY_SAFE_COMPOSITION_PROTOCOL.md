# N25 Complementary Safe Composition Protocol

Status: **preregistered before candidate construction or new official-engine games**  
Date: 2026-08-01 (America/Denver)  
Protocol: `rocket-n25-complementary-safe-composition/v1`

## Hypothesis and selection disclosure

N15's localized exact-Rocket counterfactual residual and N8B's
Recovery-minus-V110 action-head delta act through disjoint parameter paths. The
residual supplies deck-native targeted corrections; the head delta supplies a
small opponent-agnostic direction derived from two same-deck Marnie sibling
policies. N25 tests one exact additive composition:

```text
N25 = N15 terminal residual
      + 1.0 * (N8B delta_100 head - live head)
```

There is no scale sweep, checkpoint choice, interpolation, retraining, or
second candidate.

This hypothesis was motivated by prior development-panel signs: N15 was strong
against V110 but one game negative against RecoveryV4, while N8B delta 1.0 was
slightly positive against both. That is selection information. Consequently,
an N25 development pass can only authorize a large, fresh, candidate-only
confirmation; it cannot itself establish strength or deployment authority.

## Frozen sources

- qualifying Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- zero-residual base SHA-256:
  `0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb`;
- residual runtime SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`;
- N15 terminal model SHA-256:
  `354a41f5d4146ae2b48b3935653a0e47f535150eebcd37e90aa866d98d0c5ee7`;
- N15 training report SHA-256:
  `b508381af55d3f86654ab527622bec333696fa7ab37b05fb06f5b71933d92ce1`;
- N8B delta-1.0 model SHA-256:
  `1ef2a93c982072d8fca105c0b80686885338b94077a98d5227163e6a4d62fede`;
- N8B arms manifest SHA-256:
  `3eccbc2b4d10138dac66391e5ac83e18a4e3d951c920adef0fbfb7d14aebccd2`.

The builder must prove that N8B delta-1.0 differs from the live model in only
`logit_head.weight` and `logit_head.bias`. N25 must equal N15 in every other
tensor and equal N15 plus that exact float32 head delta in those two tensors.

## Offline admission

Use only the frozen N6R exact-feature supports and N9 exact-Rocket replay
supports. No prior or new match outcome may enter construction or admission.

The sole candidate advances to packaging only if all of:

1. exact source hashes and exact two-head-tensor composition pass;
2. serialization round-trip is tensor-exact;
3. only `logit_adapter.*`, `logit_head.weight`, and `logit_head.bias` differ
   from the zero-residual live base;
4. all 48 strict N6R teacher actions are top-1, with minimum margin at least
   0.25;
5. N6R non-target retention is at least 99.0% overall and 98.5% in each
   opponent stratum, with mean forward KL at most 0.02;
6. N9 retention is at least 98.0% overall and 98.0% in each opponent stratum,
   with mean forward KL at most 0.01;
7. all metrics use the full composed model path, including both the changed
   head and residual, after deployed legality/attack-END guards.

These gates deliberately permit at most a small composition-level expansion
beyond the older single-direction gates. Failure ends N25; no scale adjustment
is allowed.

## Runtime admission

If offline admission passes, construct one residual-capable candidate bundle
and a two-arm manifest containing exact live control and N25. Before match
collection require:

- exact deck and all frozen source/runtime hashes;
- packaged `NNAgent` versus strict in-process greedy agreement on at least 600
  official-engine Team Rocket decisions across both opponents;
- zero mismatches, zero illegal actions, zero exception fallbacks, and exact
  model/bundle serialization hashes.

## Locked development geometry

Only after runtime admission, freeze a separate execution lock and collect
four new uninterrupted shards with Rocket in seat 1:

- V110 seeds `825101` and `825102`;
- RecoveryV4 seeds `825201` and `825202`.

Each shard contains 120 control and 120 candidate games in a seeded shuffled
schedule. Total: 960 strict official-engine games, 240 per arm/opponent cell.
Outcomes remain unopened until all four shards seal.

Analyze 20,000 equal-opponent bootstrap replicates under seed `825001`.
Development advancement requires:

1. strict correctness and zero fallbacks;
2. at least +3.0 percentage points equally stratified;
3. strictly positive point improvement against each opponent; and
4. stratified 95% lower bound greater than -0.5 points.

## Confirmation and deployment boundary

A development pass authorizes only a separately preregistered, uninterrupted
fresh confirmation of at least 3,200 games, with new seeds and no N25
development rows, followed by both-seat/broad-regression, package audit, and
arena submission gates. Because the candidate was composed from selected prior
directions, confirmation must require positive aggregate and worst-opponent
effects with a lower confidence bound above zero.

The engine transitions and legality checks are genuine official execution.
The old counterfactual target set and parameter composition are research
abstractions. No hidden opponent hand, prize identities, seed, opponent name,
or replay label is a runtime input. No local pass is a rank-one claim. The live
bot remains unchanged until every later gate passes.

