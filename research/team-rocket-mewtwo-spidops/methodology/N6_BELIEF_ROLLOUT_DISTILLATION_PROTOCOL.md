# N6 Belief-Rollout Distillation Protocol

Status: preregistered development protocol. This document fixes the next
strength campaign before its development results are inspected. It does not
authorize arena submission.

## Why N6 exists

The live Rocket policy is a broad behavior-cloned relational network with a
small amount of later policy optimization. It is not a trained realization of
the six-phase belief-aware planner. Its largest measured weakness is the
second-seat Marnie matchup. N5 tried to improve that matchup with terminal PPO,
but none of its four independently trained arms beat the frozen control with a
positive confidence bound.

The earlier counterfactual teacher was also too noisy to use. On the same 95
roots, independent three-particle searches selected the same nominal best
action only 22 times; both declared a robust intervention nine times and agreed
on that intervention only four times. N6 therefore changes the learning target
rather than scaling the failed N5 update.

## Frozen substrate

- Deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Initial policy SHA-256:
  `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`.
- Initial policy directory:
  `artifacts/team_rocket_recovery/candidates/scale_value_calibrated_v1`.
- Development opponents are the frozen extracted bundles
  `AshMarnieGeneralV110_extracted` and
  `HelbrosMarnieGeneralRecoveryV4_extracted`.
- All source matches and descendant transitions execute through the bundled
  official `cg` engine.

## Teacher boundary

N6 is an offline belief-sampled rollout teacher, not online sampled-rollout
search.

At the first eligible Rocket main-phase choice on each retained early turn:

1. The frozen neural blueprint supplies a semantic top-three action shortlist.
2. The collector samples 12 joint hidden-zone allocations consistent with the
   acting player's observation and the frozen 60-card deck priors. It never
   reads the visualizer or the realized opponent hand.
3. Each allocation is evaluated with two common-random coin streams. Every
   shortlisted action receives the same allocation and coin stream.
4. After the forced root action, both players use frozen greedy neural
   continuation policies until a genuine official-engine terminal result.
5. Only complete paired terminal sets enter a teacher target. Leaf-value
   substitution is forbidden in N6.

The official engine genuinely executes each descendant. The distribution over
hidden allocations is a project abstraction, not an official posterior, an
equilibrium, or a safe-strategy certificate.

## Development collection

- Seat: Rocket starts second.
- Root window: turns 1 through 8.
- Roots: at most two distinct turns per accepted game.
- Candidates: three semantically distinct actions including the blueprint.
- Samples: 12 hidden allocations x 2 paired coin streams = 24 per action.
- Maximum rollout decisions: 600.
- Per-root fail-closed wall-clock cap: 600 seconds. A root reaching the cap is
  incomplete and cannot enter a teacher target.
- Required completeness: every candidate must terminate for every retained
  pair; partial pairs are excluded and tracked.
- Collector SHA-256:
  `fc1fab97dbceee170182417df93eecfd11059747020f9ce70f1e29ce681c065b`;
  inherited official-causal runtime helper SHA-256:
  `8309391b47fc39bd869d1c4d31e2af51e2f46af04d1305cc8c891d1cde85a693`.
- Training: four 25-game shards per opponent (100 accepted games each).
  V110 seeds are `790111`, `790112`, `790113`, `790114`; Recovery seeds are
  `790211`, `790212`, `790213`, `790214`.
- Selection: three fresh 20-game shards per opponent (60 accepted games each).
  V110 seeds are `790121`, `790122`, `790123`; Recovery seeds are `790221`,
  `790222`, `790223`.
- The earlier serial pilot files were interrupted before a manifest was
  produced, are retained under `work/n6_aborted_serial_*`, and are excluded
  from training, selection, validation, and every strength claim.
- A 14-process scheduling pilot showed that the former 180-second cap could
  truncate roots under CPU contention. Those files are retained under
  `work/n6_aborted_oversubscribed/` and are likewise excluded. The locked run
  schedules the eight training shards first, followed by the six selection
  shards, with the 600-second cap above; no target-quality rule changed.

The collector stores ordinary agent observations, public option summaries,
opaque particle manifests, and terminal utilities. It does not serialize raw
hidden allocations.

## Fixed target construction

For each root, utilities are compared in matched particle/coin pairs. Candidate
quality is the mean paired utility. Uncertainty is clustered by hidden
allocation: the two coin streams are averaged within allocation before
resampling allocations.

A non-blueprint action is an admissible hard teacher action only when all of
the following hold:

- it has the highest full-sample mean;
- both six-allocation halves rank it no worse than the blueprint;
- its allocation-cluster bootstrap 95% lower bound versus the blueprint is
  strictly positive; and
- at least 22 of 24 paired terminal samples are present for every shortlisted
  action.

All other roots retain the blueprint hard target. Soft teacher probabilities
may use all complete paired means, but their total policy loss weight is capped
by uncertainty and a frozen-reference KL penalty.

## Student and development arms

Search is distilled into the neural runtime. No search module is packaged in an
N6 student.

Because N6 targets the measured second-seat failure mode, any eventual packaged
candidate must use a public seat-conditioned blueprint boundary: the selected
N6 neural model only when Rocket started second, and the byte-frozen initial
model when Rocket started first. The condition uses only the official public
`firstPlayer` marker. It is not an opponent-style exploit and cannot inspect
private cards. Runtime-parity tests must prove that the first-seat branch is
exactly the frozen control action path.

The student starts from the frozen initial policy. Only the final relational
transformer block and action head are trainable. The value head is not trained
from these action comparisons. The target retains the frozen policy's total
probability mass on the shortlist but redistributes that mass by a softmax of
the official terminal means. The redistribution coefficient is
`clip(max(mean)-min(mean), 0, 1) * min(1, sqrt(complete_allocations / 12))`;
the remaining mass stays at the reference distribution.

Training is fixed at four epochs, batch size 64, AdamW learning rate `2e-5`,
weight decay `1e-4`, and seed `790301`; only the final checkpoint is eligible.
Optimization uses the residual cross-entropy
`CE(teacher, student) - CE(reference, student)` so a zero-confidence target has
exactly zero policy gradient. The frozen-reference KL has a `1e-4` numerical
dead zone; this prevents AdamW from amplifying floating-point differences when
student and reference are functionally identical.
The four arms are:

- A: teacher temperature `0.35`, reference KL coefficient `1.0`;
- B: teacher temperature `0.50`, reference KL coefficient `1.0`;
- C: teacher temperature `0.75`, reference KL coefficient `1.0`;
- D: teacher temperature `0.50`, reference KL coefficient `2.0`.

A fifth arm is the untouched control.

Candidate selection uses only selection-shard teacher metrics and a fresh
official-engine development match panel. The panel has four independently
scheduled shards: V110 seeds `791101` and `791102`, Recovery seeds `791201`
and `791202`, with 120 games per arm in each shard. Thus each arm receives 240
games per opponent and 480 games in aggregate. The fixed analysis uses 20,000
opponent-stratified independent-game bootstrap replicates with seed `791001`.
An arm advances only if it:

- has no legality or runtime-parity failure;
- improves aggregate paired win rate over control by at least 1.5 percentage
  points;
- has a positive worst-opponent point estimate; and
- has a stratified game-bootstrap 95% lower bound above -0.5 percentage
  points.

If no arm passes, N6 stops without opening a validation panel.

## Integrity-only hardening log

On 2026-07-29, after the training corpus and its aggregate analysis had closed
but before any selection outcome was opened, the downstream analyzer and
trainer were hardened to fail closed on malformed evidence. They now require a
matching companion-manifest hash, official descendant execution, no realized
hidden-truth access, no serialized hidden allocations, 24/24 paired samples,
three candidates, 12 particle manifests, zero rollout errors, lossless feature
encoding, and disjoint train/selection observation fingerprints. A regression
test also proves that post-manifest JSONL mutation is rejected.

This hardening did not change the collector, hidden-allocation sampler,
estimator, bootstrap rule, teacher target, training hyperparameters, arm set,
or advancement gates. The running selection collectors had already imported
their frozen collection code and were not modified. Current downstream hashes
are:

- analyzer: `5d00e5e24c5d61c005d6247f86e64a4db340f956c58a48e1acee1c5112711fff`;
- trainer: `219f1fb36dd20d89239dce88ece391fcffd5e1130b92e685f5afebfeb38812eb`;
- guard tests: `cb68946d2ee733b6713b671a86cfb8d9167bd61cd66e6eff2291a20caa06b829`.

These are the historical N6 downstream hashes at the time of that hardening.
After N6 distillation was invalidated on non-lossless persistent-belief feature
reconstruction, the trainer and analyzer were generalized for the separately
preregistered N6R v2 schema. Their N6R hashes and the exclusion of all N6 rows
from training are recorded in `N6R_EXACT_FEATURE_BELIEF_ROLLOUT_PROTOCOL.md`
and `N6_BELIEF_ROLLOUT_RESULT.md`.

## Later validation and deployment boundary

Any advancing arm must be frozen before a new validation seed schedule is
generated. Validation, regression, packaging, and arena submission require
separate immutable lock documents. No N6 development result can authorize a
submission. No Kaggle submission is in scope.

## Relationship to the six-phase plan

N6 operationalizes Phase 3 and Phase 5 at a tractable boundary: public-belief
particles and complete official continuations create search-improved targets,
then the policy absorbs them. It does not yet provide equilibrium training,
counterfactual-value safety bounds, or calibrated hidden-hand beliefs. Those
remain required for the durable neural game-theoretic target architecture.
