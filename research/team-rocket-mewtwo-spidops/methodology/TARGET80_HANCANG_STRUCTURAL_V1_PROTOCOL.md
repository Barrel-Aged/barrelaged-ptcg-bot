# Target-80 Hancang structural deck V1 protocol

Status: **completed and closed at Stage A**  
Locked: 2026-08-05

The panel passed integrity but failed the frozen strength gates. See
`TARGET80_HANCANG_STRUCTURAL_V1_RESULT.md`.

## Hypothesis

Test whether an exact public Wobbuffet-free Rocket deck observed in two selected
top Kaggle episodes is a materially stronger substrate than the flg deck used
by V6 when both execute the same frozen ScaleBC neural policy and the same two
deployment guards.

The public list replaces one Mimikyu, one Poffin, Archer, one Giovanni, and two
Proton with an eighth Grass Energy, two Night Stretcher, a second Brave Bangle,
Petrel, and a fourth Lillie's Determination. This is a coherent recovery and
backup-attacker package, not a one-card swap. The two selected public wins are
provenance only and are not treated as unbiased strength evidence.

## Frozen arms

Control:

- path: `work/target80_candidates/flg_wobbfree_scale_control_v1`;
- deck SHA-256:
  `e2f02e3d29a37c3c126daf92c920693a6d25ec00943e2ee725422dbf27604b73`;
- entrypoint SHA-256:
  `bb58758fa7b6b0f974bd6766c11e1cd8eb304ef834e07c4105924a0af386ca24`.

Candidate:

- path: `work/target80_candidates/hancang_wobbfree_scale_control_v1`;
- exact public ordered deck SHA-256:
  `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`;
- entrypoint SHA-256:
  `3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341`;
- public source episodes: `89819570`, `89819618`.

Both arms use model SHA-256
`3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`
and identical `soldier/` code, deck library, early-Spidops Poke Pad guard, and
Brave Bangle lethal guard. Both contain zero Wobbuffet. The only behavioral
difference is the exact deck supplied to the policy and official engine.

Collector SHA-256:
`dd7ee1f8a352321eddfcb1a10016191ea0e8f3e0b90a7b28d16fa2c19b5e8171`.

## Locked Stage-A panel

Every row interleaves eight candidate and eight control games in the bundled
official engine. Both entrypoints and the opponent execute as isolated arena
workers with strict legality. Run all rows before opening outcomes.

| Opponent | First seed | Second seed |
|---|---:|---:|
| `AshV178MarnieFullBest_20260801_extracted` | 806001 | 806002 |
| `SparkStarmieFroslassCFV13_extracted` | 806011 | 806012 |
| `SparkStarmieMistFlatV20_20260801_extracted` | 806021 | 806022 |
| `SparkRocketSpidopsV1_extracted` | 806031 | 806032 |
| `SoldierPureRLV45_extracted` | 806041 | 806042 |
| `VioletAlakazamMKV93_extracted` | 806051 | 806052 |
| `SparkTrueCrustleCrispinGalvV71_extracted` | 806061 | 806062 |

`first` and `second` denote the Rocket arm's fixed starting order. Total:
224 games, 112 per arm, 16 candidate games per opponent, and 56 candidate
games per start order.

## Integrity and Stage-A admission

Reject on any changed hash, nonterminal game, illegal action, timeout, worker
failure, missing row, wrong schedule count, Wobbuffet, or hidden-information
use. Draws count as non-wins.

The structural list advances only if it satisfies every condition:

1. at least 84 candidate wins in 112 games (`75%`);
2. at least six more wins than control (`+5.36` points);
3. at least 40/56 wins from each start order (`71.4%`);
4. at least 8/16 wins against every opponent; and
5. zero execution or legality failures.

These deliberately high gates are a plausibility screen for the user's 80%
target, not a claim that 112 development games prove 80% strength. Failure
closes this exact deck under the frozen transferred policy; thresholds may not
be weakened after opening.

If every Stage-A gate passes, freeze one candidate and create a new untouched
200-game protocol requiring at least 160 wins before any internal-arena
submission. V5 and V6 remain immutable, identities may not be pooled, and
Kaggle submission is out of scope.
