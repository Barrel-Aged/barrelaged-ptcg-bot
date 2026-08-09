# Target-80 state-conditioned plan selector V4 protocol

Status: **prospectively locked corpus and validation experiment; no deployment authority**  
Locked: 2026-08-05, before any V4 corpus shard.

## Authorization and hypothesis

The preregistered V3 own-turn milestone passed all seven gates. Its 16 fresh
roots produced five stable non-baseline plan changes, both start orders were
represented, particle-half and coin-stream stability were each 75%, and the
mean paired dense gain over stable changes was +0.310. V4 uses that narrowly
granted authority to test whether a conservative state-conditioned selector
can predict such stable plan changes on game-disjoint own-turn roots.

V4 does not authorize a runtime, package, arena submission, or strength claim.
A pass authorizes only a new, separately frozen selector test corpus. The
selector must subsequently pass untouched plan-label testing, full-match
official-engine development, and untouched full-match confirmation before it
can be considered for packaging.

## Frozen substrate

The exact 60-card Rocket package remains Wobbuffet-free:

- tree:
  `30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f`;
- deck:
  `4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea`;
- model:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- main:
  `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`.

The Marnie opponent tree is
`49bb3353ab42fded18e32e38f154ad1a001d60c5852383045ff4f437be36efe1`;
its deck, model, and main hashes are respectively
`92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d`,
`98072b2a75197ad9d073af3c7ba5ff66a6daadd266b86b5463fef2ff34910757`,
and
`0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b`.
The search runtime tree remains
`2fab4c1c945dc97269f147b6237d37cb653228b6792263f28df05945e0744b61`.

## Frozen teacher and labels

Root selection, plans, action support, public guards, strategic axes,
state-conditioned axis aggregation, and the end-of-own-turn boundary are
unchanged from V3. The three plans are:

- `baseline`: exact guarded neural policy;
- `engine_resilience`: Spidops engine 1.25, second attacker 1.25, Charging Up
  0.75, bench discipline 0.50, fuel preservation 0.50, attack tempo 0.25;
- `tempo_conversion`: attack tempo 1.25, fuel preservation 0.75, second
  attacker 0.50, tool assignment 0.50, bench discipline 0.50.

Every retained root has at least two distinct plan actions; every
non-baseline root action has frozen policy support at least 0.01. Exact
Early-Pad-to-Spidops and lethal-only Brave-Bangle public guards always retain
priority. Each root is evaluated with eight public-consistent hidden-zone
particles and two shared coin streams: 16 matched sets and 48 plan lines.

The teacher selects a non-baseline label only when all of the following hold:

1. the same non-baseline plan wins the full sample, both four-particle halves,
   and both coin streams;
2. its full mean paired dense gain over baseline is at least 0.05;
3. its paired gain is nonnegative in each particle half and coin stream.

Every other root is labeled `baseline`. Terminal value remains four times
match utility and dominates the bounded dense milestone score.

## Locked corpus

All roots are fresh and game-disjoint from V1-V3. One highest-priority
eligible critical root is retained per source game, turns 2-8, after inspecting
at most eight candidate roots. Limits are 1,200 source attempts, 2,000 live
decisions, 150 branch decisions, 180 seconds per root, and one Torch thread.

| Shard | Split | Rocket start order | Roots | Seed | Output |
|---|---|---|---:|---:|---|
| `train_first_a` | train | starts first | 40 | 859101 | `work/target80_plan_selector_v4_corpus/train_first_a.jsonl` |
| `train_first_b` | train | starts first | 40 | 859102 | `work/target80_plan_selector_v4_corpus/train_first_b.jsonl` |
| `train_second_a` | train | starts second | 40 | 859201 | `work/target80_plan_selector_v4_corpus/train_second_a.jsonl` |
| `train_second_b` | train | starts second | 40 | 859202 | `work/target80_plan_selector_v4_corpus/train_second_b.jsonl` |
| `validation_first` | validation | starts first | 24 | 859111 | `work/target80_plan_selector_v4_corpus/validation_first.jsonl` |
| `validation_second` | validation | starts second | 24 | 859211 | `work/target80_plan_selector_v4_corpus/validation_second.jsonl` |

All four training shards and both validation shards must seal before any plan
ranking, label distribution, or outcome is inspected. Progress counts and
manifests may be observed only for execution integrity. A changed hash,
missing root, incomplete paired set, rollout error, guarded-baseline mismatch,
hidden-hand/visualizer read, raw hidden allocation, opponent action inside the
scoring horizon, or duplicate shard/root invalidates the corpus.

## Frozen selector

The input is an 834-dimensional vector, standardized using training statistics
only. It contains the serialized public/own-visible neural policy feature,
padded board entities, public/own-visible belief summaries, each plan's root
action feature and frozen policy score, start order, turn, policy margin, and
criticality flags. It contains no teacher value, match result, future event,
visualizer output, or realized hidden allocation.

Three fixed MLPs are trained with AdamW, batches of 32, gradient norm clipping
at 2.0, cross-entropy label smoothing 0.03, and square-root inverse class
weights clipped to [0.5, 3.0]:

| Name | Hidden | Dropout | Weight decay | LR | Epochs | Seed |
|---|---:|---:|---:|---:|---:|---:|
| conservative | 96 | 0.10 | 0.001 | 0.001 | 120 | 860101 |
| balanced | 128 | 0.10 | 0.0003 | 0.001 | 150 | 860102 |
| capacity | 192 | 0.15 | 0.0001 | 0.0007 | 150 | 860103 |

At inference, the highest-probability non-baseline plan is selected only when
its probability is both greater than baseline and at least one frozen threshold
in {0.55, 0.65, 0.75, 0.85}; otherwise the exact baseline is retained.
Configuration and threshold selection use the locked validation metrics below,
ordered by mean paired teacher gain over all rows, then precision, then lower
change rate. There is no post-result threshold or architecture retuning.

V4 requires at least 20 stable non-baseline labels among the 160 training
roots. A configuration/threshold passes validation only if all gates hold:

1. change rate is 5%-30%;
2. exact non-baseline-plan precision is at least 65%;
3. exact non-baseline-plan recall is at least 20%;
4. stable-baseline retention is at least 90%;
5. mean paired teacher gain across all validation rows is at least 0.04;
6. at most 15% of changed rows have a negative paired teacher gain;
7. mean paired teacher gain is nonnegative for both start orders.

A pass authorizes only a fresh, game-disjoint selector test collection under a
new protocol. Failure closes this exact V4 learner without inspecting a test
set or selecting a looser threshold.

## Claim boundary

Source games, descendant transitions, and action legality are genuine official
engine execution. Hidden particles, public-belief summaries, persistent plans,
six-axis milestones, meta weights, and the learned selector are research
abstractions. The teacher is sampled offline causal supervision, not the final
strategic approach. V4 neither models an opponent response within its scoring
horizon nor establishes equilibrium safety or match-win improvement.

## Frozen implementation

- collector:
  `ec0dac8c4943e773ac325b41f333ad1f56dbbefb5f02e598af37c96e8d56d710`;
- trainer/gate:
  `00560dc4db4f897af19d22eda57c3509bca2e7871c9ec778e89d45d68326c11b`;
- V3 collector dependency:
  `4c2afa3f85d279b63ac431702fae90fbb2b2ba9e277349529b55fd162cd35860`;
- V2 collector dependency:
  `8e153f1baceb240ac522275aea869e170ce9bfdd2c0ad25ae4bb4118c068da17`;
- V2 analysis dependency:
  `4b458e5c14c261c83c6b41f66f3391d58a5c1ab9cdfc3dd94e85a9eb4c3f2775`.

The archived V3 rows used only for the pre-lock 834-dimensional vector
contract smoke are excluded from all V4 training, validation, testing, and
strength claims.
