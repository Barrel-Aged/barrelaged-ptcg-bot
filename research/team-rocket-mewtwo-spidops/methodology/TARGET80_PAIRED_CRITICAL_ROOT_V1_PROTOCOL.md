# Target-80 paired critical-root V1 protocol

Status: **prospectively locked pilot; no deployment authority**  
Locked: 2026-08-05, after one excluded integration smoke and before either
pilot shard is collected.

## Purpose

The 1,200-game randomized-support corpus repaired deterministic action support,
but Randomized Advantage V1 and Residual Advantage V2 showed that one terminal
label per trajectory is too weak for per-decision credit assignment. This pilot
tests the next queued route: directly compare plausible actions from the same
strategically important public root under matched official-engine
continuations.

This is deliberately not another broad distillation of N6R's first two early
MAIN decisions. It uses the current Wobbuffet-free package, keeps one
highest-priority critical root per source game, covers both seats against the
current hard Marnie opponent, and tests ranking stability before authorizing a
larger corpus or learner.

## Frozen substrate

The Rocket package is
`work/target80_candidates/mimikyu_maxbelt_neural_control_v1`:

- bundle tree SHA-256:
  `30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f`;
- deck SHA-256:
  `4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea`;
- model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- main SHA-256:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`;
- deck integrity: exactly 60 cards and zero Wobbuffet (card ID 432).

The source actor and every continuation preserve both packaged public guards:
the turn-four-or-earlier Poke Pad selection of a missing Spidops evolution and
the lethal-only Brave Bangle attachment. Every retained root must reproduce the
guarded baseline action from a copied actor-visible belief state.

The opponent is
`work/arena_bundles/AshV178MarnieFullBest_20260801_extracted`:

- bundle tree SHA-256:
  `49bb3353ab42fded18e32e38f154ad1a001d60c5852383045ff4f437be36efe1`;
- deck SHA-256:
  `92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d`;
- model SHA-256:
  `98072b2a75197ad9d073af3c7ba5ff66a6daadd266b86b5463fef2ff34910757`;
- main SHA-256:
  `0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b`.

The official search-capable runtime is
`artifacts/team_rocket_recovery/candidates/scale_neural_resolver_safe_tie_v3`,
tree SHA-256
`2fab4c1c945dc97269f147b6237d37cb653228b6792263f28df05945e0744b61`.

## Genuine execution and abstractions

Genuine official-engine facts:

- every source game is executed through official `cg` battle transitions;
- every candidate branch is created through official `search_begin` and
  `search_step` and continued through official transitions to a terminal game;
- legality comes only from the official observation and legal-option list;
- the exact packaged Rocket guards execute in both source games and branches;
- common random coin sequences are shared across candidates inside a matched
  particle/coin set.

Research abstractions:

- hidden-zone completions are sampled from the redacted public observation and
  frozen deck priors; they are not an official posterior or realized truth;
- the opponent continuation is the frozen neural blueprint represented by its
  model, not a claim about every possible arena opponent strategy;
- eight particles are a small pilot approximation, not a solved information
  set or counterfactual-value certificate;
- terminal rollout averages are offline teacher evidence only. Sampled rollout
  search is not the intended final strategic policy.

The visualizer, realized opponent hand, and raw sampled hidden allocations are
forbidden. The opponent hand in every serialized observation must remain
`null`.

## Critical-root rule

For each eligible source game, capture at most eight distinct-turn MAIN roots
from turns 2 through 10. A root must have a single legal selection, at least two
semantically distinct candidates, and every non-baseline candidate must have at
least 0.02 probability under the frozen attack-safe policy.

A root is eligible if at least one of these public conditions holds:

- the top-two policy-logit margin is at most 1.0;
- the shortlist contains both an attack and a setup/attachment/evolution/
  ability/retreat action;
- Mewtwo is in play but not ready and the shortlist can change setup or Energy;
- either player has three or fewer prizes remaining;
- from turn four onward, Rocket lacks two ready attackers and the shortlist can
  play, attach, or evolve.

The deterministic priority score is, in order of weight: prize threshold 10,
attack/setup tradeoff 8, Mewtwo activation 6, missing second attacker 4,
sub-three-Spidops setup 3, low policy margin 2, plus `0.10 * turn`, minus the
policy margin capped at 3. Exact ties prefer later turn, later source step, then
lexicographically higher observation fingerprint. Only the highest-ranked root
from a source game is retained, preventing multiple correlated roots from one
trajectory.

## Locked collection

Two shards must finish and seal before analysis:

| Shard | Rocket start order | Retained roots | Seed | Output |
|---|---|---:|---:|---|
| first | starts first | 8 | 856101 | `work/target80_paired_critical_root_v1/pilot_first.jsonl` |
| second | starts second | 8 | 856201 | `work/target80_paired_critical_root_v1/pilot_second.jsonl` |

Every root uses:

- at most three candidates (guarded baseline plus up to two supported semantic
  alternatives);
- eight public-consistent hidden-zone particles, IDs 0 through 7;
- two common-random coin streams per particle;
- 16 fully matched sets and terminal continuation for every candidate;
- maximum 600 rollout decisions and 600 seconds per root;
- one Torch thread;
- no leaf-value substitution.

Both shards use `capture_roots_per_game=8`, alternative support 0.02, turn
window 2-10, maximum 200 source attempts, and maximum 2,000 live decisions.
Failure to retain all eight roots, any illegal/fallback branch, any rollout
error, incomplete matched set, feature/guard parity failure, hidden-hand leak,
or changed hash invalidates the shard. Failed roots may not be silently dropped
or replaced after their results are read.

## Locked stability analysis and admission gates

Outcomes remain unopened until both manifests exist and pass their integrity
checks. The analyzer independently reproduces full, particle-half, and
coin-stream action rankings from the 16 paired samples per root.

A root is a stable non-baseline change only if:

- full, first-particle-half, second-particle-half, first-coin-stream, and
  second-coin-stream rankings all choose the same non-baseline action;
- its full paired mean improvement over baseline is at least 0.125 terminal
  utility;
- both particle-half and both coin-stream paired mean deltas are nonnegative.

The pilot passes only if all gates pass:

1. all 16 roots and all 256 matched particle/coin sets are complete, with zero
   errors and full guarded-baseline reproduction;
2. at least 12/16 roots have stable full/particle-half rankings;
3. at least 12/16 roots have stable full/coin-stream rankings;
4. at least three roots are stable non-baseline changes;
5. each seat contributes at least one stable non-baseline change;
6. the mean paired delta across stable changes is at least 0.125;
7. retained roots cover at least three criticality categories.

Passing authorizes only an expanded, game-disjoint paired-root collection and a
separately locked learner. It does not authorize a runtime overlay,
development matches, confirmation, packaging, or submission. Failure closes
this exact pilot; thresholds may not be loosened after seeing results.

## Frozen implementation

- collector:
  `4f6d26ff7aef0b79341e589c79321a883f1201a6a4b2faf8d6fe4539e92d94dd`;
- analyzer:
  `1adf7acf8d221757fea1282d537e5af3aa3cb59acf16a0be363ba4d97f738d2f`;
- inherited public-belief rollout runtime:
  `fc1fab97dbceee170182417df93eecfd11059747020f9ce70f1e29ce681c065b`;
- inherited official causal helper:
  `8309391b47fc39bd869d1c4d31e2af51e2f46af04d1305cc8c891d1cde85a693`;
- typed actor-feature codec:
  `4b76b88729621bea1b2c621283e6616de56f11a5810a803e22d2623cc2972906`.

The one-root integration smoke at
`work/target80_paired_critical_root_v1_smoke/` used seed 856001, one particle,
and one coin stream. It verified official branch execution, guard-aware feature
parity, hidden-boundary serialization, and manifest creation. It is permanently
excluded from pilot, training, development, confirmation, and strength claims.
