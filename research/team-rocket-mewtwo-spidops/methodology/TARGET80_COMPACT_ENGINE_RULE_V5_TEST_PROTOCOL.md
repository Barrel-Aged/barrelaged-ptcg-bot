# Target-80 compact engine-resilience rule V5 test protocol

Status: **prospectively locked fresh paired test; no deployment authority**  
Locked: 2026-08-05, before either V5 test shard.

## Motivation and development boundary

The exact V4 834-feature selector failed held-out admission and is closed. Its
paired corpus is now development evidence, not an untouched test. Across its
208 roots, 51/54 stable non-baseline labels chose `engine_resilience`. A single
public, interpretable pattern accounted for a useful high-value subset:

1. the engine plan's root action attaches Grass, Psychic, or Team Rocket's
   Energy to a Team Rocket's Spidops that currently has exactly one energy; and
2. the baseline action either has no public in-play target or targets a
   zero-energy Pokemon other than Team Rocket's Tarountula.

On the openly analyzed V4 development corpus, this frozen rule changed 38/208
roots, with 68.4% exact-plan precision, 48.1% exact-plan recall, 92.9% baseline
retention, +0.2545 mean paired gain per changed root, +0.0465 per development
root, 10.5% negative changes, and positive mean gain in both start orders.
Those exploratory numbers selected the rule and are not confirmation or a
strength claim.

V5 tests the rule without fitting, thresholds, or alternatives on fresh,
game-disjoint paired roots. Failure closes this exact rule. Passing authorizes
only full-match official-engine development integration under a new protocol.

## Frozen substrate and causal teacher

The exact Wobbuffet-free 60-card candidate remains frozen:

- tree:
  `30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f`;
- deck:
  `4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea`;
- model:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- main:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`.

The Marnie opponent tree remains
`49bb3353ab42fded18e32e38f154ad1a001d60c5852383045ff4f437be36efe1`;
the search runtime tree remains
`2fab4c1c945dc97269f147b6237d37cb653228b6792263f28df05945e0744b61`.

Root selection, baseline guards, three plan vectors, six milestone axes,
state-conditioned aggregation, 0.01 plan-action support floor, and the
end-of-own-turn scoring boundary are unchanged from V3/V4. Every retained root
uses eight public-consistent hidden-zone particles and two common coin streams:
16 matched sets and 48 plan lines.

The stable teacher is also unchanged. A non-baseline label requires the same
winner over the full sample, both four-particle halves, and both coin streams;
full paired gain at least 0.05; and nonnegative paired gain in every half and
stream. Otherwise the label is baseline. Terminal value dominates the bounded
dense score.

## Frozen rule

The rule may inspect only the normal redacted observation and plan root actions.
It selects `engine_resilience` exactly when:

- the engine action type is ATTACH;
- its own-visible hand card is ID 1, 5, or 15;
- its public target is card ID 401 with exactly one attached energy; and
- the baseline action has no public board target, or its target has zero energy
  and is not card ID 400.

Every other state selects the exact guarded baseline. There is no learned
checkpoint, score threshold, fallback rule, or `tempo_conversion` selection.

## Fresh locked test

| Shard | Rocket start order | Roots | Seed | Output |
|---|---|---:|---:|---|
| `test_first` | starts first | 48 | 861111 | `work/target80_compact_engine_rule_v5_test/test_first.jsonl` |
| `test_second` | starts second | 48 | 861211 | `work/target80_compact_engine_rule_v5_test/test_second.jsonl` |

Root turns are 2-8; at most eight candidate roots are inspected per source
game. Fixed limits are 1,200 source attempts, 2,000 live decisions, 150 branch
decisions, 180 seconds per root, and one Torch thread.

Both shards must seal before any label, plan value, ranking, or outcome is
inspected. Progress and manifest integrity counts may be observed. A changed
hash or identity, fewer than 96 complete roots/1,536 matched sets, rollout
error, guarded-baseline mismatch, opponent action inside the scoring horizon,
visualizer/realized-hidden read, or raw hidden allocation invalidates the test.

## Locked admission gates

V5 passes only if all gates hold on the combined fresh test:

1. all 96 roots and 1,536 matched sets are integrity-complete;
2. rule change rate is 5%-30%;
3. exact non-baseline-plan precision is at least 65%;
4. exact non-baseline-plan recall is at least 20%;
5. stable-baseline retention is at least 90%;
6. mean paired teacher gain across all roots is at least +0.04;
7. at most 15% of changed roots have negative paired gain;
8. mean paired gain is nonnegative in both start orders.

There is no post-test rule amendment or gate relaxation. A pass grants only
full-match development integration. That integration must preserve the exact
public guards and plan persistence, and must beat an identical-deck guarded
baseline across both seats and a broad opponent panel before untouched
confirmation, packaging, or arena submission can be considered.

## Claim boundary

Source games, descendant transitions, and legality are genuine official-engine
execution. Hidden particles, persistent plans, six-axis milestones, meta
weights, stable teacher, and compact rule are research abstractions. The paired
test measures an own-turn causal milestone, not match win rate, opponent
response, equilibrium safety, or arena strength. Sampled rollouts remain
offline supervision and are not the final strategic approach.

## Frozen implementation

- collector:
  `fc6a3eddf3065c2334c47178c9a01d4a55b91cf9a6c358591d89af342f53ee89`;
- analyzer/gate:
  `1ab324719fb81367e7d6dbe5f7d1e49e88473c59d9318218c99bcbdf253e210d`;
- V3 dependency:
  `4c2afa3f85d279b63ac431702fae90fbb2b2ba9e277349529b55fd162cd35860`;
- V2 dependency:
  `8e153f1baceb240ac522275aea869e170ce9bfdd2c0ad25ae4bb4118c068da17`.

Hash comparison is case-insensitive. The V1-V4 corpora are excluded from this
fresh test and may not be pooled with it for admission.
