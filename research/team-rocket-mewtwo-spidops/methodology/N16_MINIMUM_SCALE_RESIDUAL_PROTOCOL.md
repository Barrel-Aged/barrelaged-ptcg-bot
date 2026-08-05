# N16 Minimum-Scale Residual Projection Protocol

Status: **preregistered before scale calculation, N16 model construction, or
match evaluation**  
Protocol: `rocket-n16-minimum-scale-residual-projection/v1`  
Preregistered: 2026-08-01

## Hypothesis and separation from N15 outcomes

N15's terminal residual passed every offline safety gate and had a positive
aggregate development point estimate, but failed its RecoveryV4 and uncertainty
gates. N16 does not extend, resample, or reanalyze that selected panel. It tests
a distinct policy constructed without reading any N15 match row or outcome:
preserve the trained residual direction, then use the smallest global scale that
keeps every one of the 48 frozen counterfactual corrections at a positive
margin.

The hypothesis is that N15's full residual magnitude includes avoidable
non-target action changes. An analytic projection can retain the complete
teacher correction set while minimizing that incidental displacement. The scale
is determined only by frozen offline logits; no scale sweep, validation outcome,
or opponent-identity router is permitted.

## Frozen inputs

- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- live/base model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- N14 zero-base model SHA-256:
  `0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb`;
- residual runtime SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`;
- N15 terminal model SHA-256:
  `354a41f5d4146ae2b48b3935653a0e47f535150eebcd37e90aa866d98d0c5ee7`;
- N15 training report SHA-256:
  `b508381af55d3f86654ab527622bec333696fa7ab37b05fb06f5b71933d92ce1`;
- N14 zero-parity report SHA-256:
  `c8638bd6d5a879d32eb88bba30f7ddeb334d222e9b295c64f0ca2619096611d5`;
- N12 transfer audit, N6R corpus, N9 parity report, and N9 replay tree: the exact
  hashes transitively frozen and checked by the N14/N15 support loaders.

The N15 development rows and analysis are forbidden inputs to the N16 builder.
The builder may not accept a scale argument.

## Deterministic analytic projection

For each strict correction state `c`, teacher-best legal option `t`, and every
other legal option `j`, let:

- `b(c,i)` be the frozen live baseline logit;
- `r(c,i)` be the N15 terminal residual logit, so the N15 logit is
  `b(c,i) + r(c,i)`;
- `a` be one global scale applied to every residual logit.

N16 requires a teacher margin `m = 0.02`:

```text
b(c,t) - b(c,j) + a * (r(c,t) - r(c,j)) >= m
```

Every inequality yields a lower bound, upper bound, or infeasibility condition
on `a`. Intersect all constraints with `[0, 1]`. Let the maximum lower bound be
`L` and minimum upper bound be `U`. The one permitted scale is:

```text
a = ceil((L + 1e-6) * 1e6) / 1e6
```

Construction fails if the interval is empty, `a > U`, `a >= 1`, or a fresh
float32 forward pass does not give all 48 teacher actions at margin at least
`0.02 - 1e-6`.

The projection is implemented by multiplying only the trained adapter's final
`16 x 1` weight by `a`; its runtime config remains scale 1.0. All other N15
terminal tensors must be exactly equal. This is algebraically the same residual
direction at one global magnitude and requires no runtime branch.

## Offline admission gates

After a fresh reload, N16 is admitted only if all of the following pass:

- the analytic constraint audit is feasible and the scale is strictly between
  zero and one;
- 48/48 teacher actions are top-1 and each has margin at least `0.02 - 1e-6`;
- only the final adapter projection differs from N15, by exactly the recorded
  global scale within float32 tolerance; all non-final tensors are equal;
- only `logit_adapter.*` differs from the zero-residual base;
- N6R non-correction top-1 retention is at least 98.0%, mean forward KL at
  most 0.05;
- N9 top-1 retention is at least 99.0% overall and at least 98.5% in each
  opponent group, mean forward KL at most 0.02;
- full runtime versus contextual-cache execution has zero top-action mismatch
  and finite relative-logit error at most `2e-5` on all 639 N6R and 6,753 N9
  actions;
- serialization is tensor-exact and the exact deck/runtime assets remain
  frozen.

Failure of any gate ends N16 before match play. Offline admission is not a
strength claim.

## Official-engine boundary

The correction labels originate from official `cg` terminal descendants under
sampled public-consistent hidden allocations. Sampling, counterfactual
estimation, cached logits, and the analytic scale projection are research
abstractions. Runtime decisions receive only public and own-visible observation
data plus the actor's persistent public-belief summary. Genuine match evidence
must come from strict official-engine execution with the packaged adapter.

## Fresh development panel if admitted

The frozen live control and sole N16 candidate each receive 240 second-seat
games per opponent, split into two 120-game shards:

- V110 seeds `816101` and `816102`;
- RecoveryV4 seeds `816201` and `816202`.

All shards must seal before outcomes are opened. Advancement requires strict
correctness, at least +2.0 percentage points equally stratified, strictly
positive improvement against both opponents, and a 20,000-replicate stratified
bootstrap 95% lower bound greater than -0.5 points under seed `816001`.

A pass authorizes only a fresh confirmation of at least 3,200 new games plus
broad both-seat regression. The live bot remains unchanged until every later
gate passes; N16 itself never directly authorizes deployment.
