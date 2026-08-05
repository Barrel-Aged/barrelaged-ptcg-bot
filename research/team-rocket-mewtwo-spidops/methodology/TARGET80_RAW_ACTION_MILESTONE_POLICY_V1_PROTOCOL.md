# Target-80 raw-action milestone policy V1 protocol

Status: **mechanism smoke passed; pilot interrupted before shard 9 by a runtime compatibility defect; no aggregate outcome inspected**

Preregistered: 2026-08-05 (America/Denver)

## Purpose

Test the next queued learning route after the zero-Wobbuffet Lightning packages
failed. This experiment supervises the native tactical policy on paired
counterfactual values for **raw legal actions** at strategically critical
official-engine roots. It is deliberately different from:

- N6/N6R terminal raw-action rollout labels, which were sparse and unstable;
- V3/V4 strategic-plan selection, which found stable plan signal but too few
  deployable selector changes; and
- hand-written runtime guards or sampled rollout search as a final policy.

The V1 pilot may authorize a larger game-disjoint corpus. It cannot authorize
training, packaging, an arena submission, or a strength claim by itself.

## Frozen source identity

Candidate:
`work/target80_candidates/hancang_wobbfree_elite_bc_v1r_selected`

- deck SHA-256:
  `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`
- selected exact-deck model SHA-256:
  `b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`
- main SHA-256:
  `3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341`
- semantic tree SHA-256 after adding source provenance:
  `1e8fb893539a14d2f9e7ff85694af0cb229e1628387fec2883e12365e53aecc6`
- exact deck size: 60;
- Team Rocket's Wobbuffet engine ID `432`: zero copies.

The model is iteration 3 from the completed exact-Hancang elite BC V1R run.
That checkpoint was rejected for direct deployment at 136-73-1 in its locked
first-seat panel. It is used only as the strongest preserved exact-deck source
policy, not treated as a validated successor.

Collector:
`scripts/collect_target80_raw_action_milestone_v1.py`

- SHA-256:
  `97685bcecd7e21072038ac698feb9cef78786b8643204cc81ef8f684be48f977`

Locked runner and aggregate analyzer:

- `scripts/run_target80_raw_action_milestone_v1_pilot.ps1`:
  `b6899ecefd7a0c50b3e29fd82af13ff29009e571efbfd75a2b808c2d00206f50`
- `scripts/analyze_target80_raw_action_milestone_v1.py`:
  `7f2bf85068fc934daf76457e54fd489ddb2173b1d5a4bacb7d520c95d2458ea0`

Any decision-affecting edit requires a new protocol/version. A purely
mechanical defect found before a valid pilot outcome may be corrected only in a
new V1R collector with the defect and hash recorded before rerunning.

## Frozen opponent panel

The pilot uses exactly seven opponents, two roots per start order per opponent
(28 roots total):

| Index | Bundle | Deck SHA-256 |
|---:|---|---|
| 1 | `work/arena_bundles/AshV178MarnieFullBest_20260801_extracted` | `92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d` |
| 2 | `work/arena_bundles/SparkStarmieMistFlatV20_20260801_extracted` | `64ef6e2c0f5fadc12f96426aa80a07e1402501ed00d1778a0badd25d171e4901` |
| 3 | `work/arena_bundles/SparkStarmieFroslassCFV13_extracted` | `60e91bf519988fd2efd14b8c9806bf03febf8baa2caab5250eab1da65b674d61` |
| 4 | `work/arena_bundles/SparkRocketSpidopsV1_extracted` | `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852` |
| 5 | `work/arena_bundles_20260804/SoldierPureRLV45_extracted` | `80798a3cae4e541da01acaa9a58c493f3d5010571e4dbabb10bf7dc701c94281` |
| 6 | `work/arena_bundles_20260804/VioletAlakazamMKV93_extracted` | `4e56a20249fad14097d06aaee2a241f355b90ef9bef19307f95b47798d0a2a3d` |
| 7 | `work/arena_bundles_20260804/SparkTrueCrustleCrispinGalvV71_extracted` | `599393da0203019c8cc3b2192c54f91ea5213d8c6e9631070df14fce1944d79c` |

The collector manifest records each full current tree and decision-file hash.
Packaged code opponents execute through the isolated official evaluator worker.

## Official execution and abstraction boundary

Genuine official-engine work:

- complete source matches;
- legal options, state transitions, chance prompts, and terminal states;
- every descendant transition after a candidate raw action.

Research abstractions:

- critical-root selection;
- top-three candidate proposal and the 2% minimum source-policy support filter;
- public-consistent hidden-zone particles sampled from frozen deck priors;
- the six strategic milestone axes and state-conditioned meta weights;
- selecting an offline teacher label from paired branch values.

No native visualizer or realized hidden truth is read. Raw sampled allocations
remain process-local. The sampler is not an official posterior, the values are
not an information-set equilibrium, and no counterfactual-value safety bound is
claimed. The opponent never acts inside the one-own-turn horizon. Sampled
branches are teacher data only and will not be the submitted strategy.

## Locked mechanism smoke

Before the pilot, run one Marnie/first-start root with seed `806099`, two
particles, one coin rollout, and the otherwise frozen settings. It must:

1. finish with one retained root;
2. reproduce the exact guarded baseline action from the serialized policy
   feature;
3. complete both particle pairs for every retained candidate;
4. produce no official-engine, worker, legality, timeout, or serialization
   error;
5. serialize no raw hidden-zone allocation.

The smoke is mechanical evidence only and is excluded from the pilot gates.

## Locked pilot collection

For each opponent, use seeds `806101/806102`, `806201/806202`, through
`806701/806702`; the odd seed is the first-start shard and the even seed is the
second-start shard. Each of the 14 shards retains exactly two roots.

Frozen settings per shard:

- capture up to eight eligible MAIN roots per source game and retain the
  highest criticality root;
- root turn window 2 through 8;
- three raw candidate actions;
- minimum alternative source-policy probability 0.02;
- eight public-consistent particles;
- two common-random-number coin continuations per particle;
- horizon: first opponent observation after the current Rocket turn ends;
- at most 150 descendant decisions and 180 seconds per root;
- Torch threads: one;
- source attempts: at most 240.

Do not inspect or tune pilot outcomes between shards. Complete all 14 locked
shards, then run one frozen aggregate analysis.

## Locked pilot advancement gates

Every gate is required:

1. All 28 roots are collected as 14 cells of exactly two roots.
2. Every root reproduces the guarded baseline and completes all 16 paired
   particle/coin samples with zero errors.
3. Candidate, opponent, collector, and dependency identities remain stable;
   every source deck is exactly 60 cards and the Rocket deck remains zero-Wobb.
4. At least 21/28 roots (75%) select the same best action in both particle
   halves and both coin slices.
5. Stable labels change the baseline at 4 through 16 roots inclusive, with at
   least one stable change in each start order.
6. Across stable changed roots, mean paired strategic advantage is at least
   `+0.06`, median root advantage is positive, and at most 25% of underlying
   paired samples are negative.
7. Mean stable-label advantage is at least `+0.03` in each start order.
8. At least five of seven opponent families have nonnegative mean
   stable-label advantage; neither Starmie family may be below `-0.02`.

Passing authorizes a separate, newly locked, game-disjoint scale-and-train
stage using native policy weights with retention constraints. Failure closes
this exact teacher configuration without threshold relaxation or post-hoc
feature changes.

No Kaggle submission is authorized. No internal-arena submission is authorized
until a trained successor passes official smoke, side-swapped broad
development, and untouched confirmation gates.

## V1 interruption

The mechanism smoke passed all five health requirements. The locked pilot then
completed eight shards and stopped before shard 9 when the older in-process
search runtime tried to construct SoldierPureRLV45's newer `v6` model and
raised `TypeError: unexpected keyword argument 'v6'`. No aggregate pilot
outcome was opened or analyzed. The eight partial shards are preserved but are
not admissible evidence and will not be mixed with the correction. V1R reruns
all 14 shards from scratch with the same seeds, parameters, and gates, changing
only the incompatible opponent's execution path to the isolated packaged
worker already used for code-only opponents.
