# Target-80 raw-action milestone policy V1R protocol

Status: **completed; admission failed; teacher configuration closed**

Outcome record: `TARGET80_RAW_ACTION_MILESTONE_POLICY_V1R_RESULT.md`

Preregistered: 2026-08-05 (America/Denver)

## Relationship to V1

This is the mechanical runtime-compatibility correction to
`TARGET80_RAW_ACTION_MILESTONE_POLICY_V1_PROTOCOL.md`. V1's Marnie mechanism
smoke passed. Its locked pilot completed eight shards, then stopped before
shard 9 because the search runtime could not construct SoldierPureRLV45's
newer in-process `v6` model. No aggregate result was opened.

V1R changes only the opponent execution path when that exact constructor error
occurs. It uses the existing isolated packaged evaluator worker, which executes
the opponent's own bundle without importing its network config into the search
runtime. All 14 shards are rerun from scratch; none of the eight V1 partial
shards is reused.

The source deck, source model, public guards, opponents, seeds, root selection,
candidate support, particles, coin continuations, strategic axes, meta weights,
horizon, sample sizes, and advancement gates remain exactly those in V1.

## Frozen identities

Source candidate:
`work/target80_candidates/hancang_wobbfree_elite_bc_v1r_selected`

- deck SHA-256:
  `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`
- model SHA-256:
  `b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`
- main SHA-256:
  `3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341`
- exact deck size 60 and Wobbuffet count zero.

V1R collector:
`scripts/collect_target80_raw_action_milestone_v1r.py`

- SHA-256:
  `5c196d4a30da6523d9d4db00a9f7aebe3b724b2dcfc40510252c855892442652`

V1R runner:
`scripts/run_target80_raw_action_milestone_v1r_pilot.ps1`

- SHA-256:
  `2ff28977d1eb34971499bcdc49abac1470789f1bf311d9a24aa2eeecf460404f`

V1R analyzer:
`scripts/analyze_target80_raw_action_milestone_v1r.py`

- SHA-256:
  `25815db1e142784a68611cd13d6e65359c35ce8697f5499ff6b1093b9dac13ec`

## V1R compatibility smoke

Before launching the fresh panel, run exactly one first-start root against
`SoldierPureRLV45_extracted` with seed `806098`, two particles, one coin
continuation, and all other V1 settings. It must report the isolated
compatibility fallback, reproduce the guarded source action, complete both
paired samples, serialize no raw hidden allocation, and have zero errors.

The smoke is excluded from strength analysis.

## Fresh locked panel

Run the V1 14-cell schedule and exact seeds `806101` through `806702` using
`scripts/run_target80_raw_action_milestone_v1r_pilot.ps1`. Each cell retains two
roots, for 28 game-distinct roots total. Do not inspect or analyze shard
outcomes until every V1R shard and manifest exists.

Apply V1 gates 1 through 8 without modification:

1. 28 roots, exactly two in every opponent/start cell.
2. Exact guarded baseline parity, all 16 paired samples per root, zero errors.
3. Stable identities, exact 60-card zero-Wobbuffet source deck.
4. At least 21/28 stable labels across both particle halves and both coin
   slices.
5. Between 4 and 16 stable changes inclusive, with both start orders present.
6. Stable changed-root mean advantage at least `+0.06`, positive median, and no
   more than 25% negative underlying paired samples.
7. Mean stable-label advantage at least `+0.03` in each start order.
8. At least five opponent families nonnegative; neither Starmie family below
   `-0.02`.

Passing authorizes only a new game-disjoint scale-and-train protocol. Failure
closes this exact teacher configuration. This remains an offline abstraction,
not an official posterior, safe information-set solve, equilibrium certificate,
or deployed sampled-rollout strategy. No Kaggle or internal-arena submission is
authorized by this pilot.
