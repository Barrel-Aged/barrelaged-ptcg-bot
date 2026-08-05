# Target-80 Maximum Belt Mewtwo/Starmie V1 protocol

Status: **completed and closed; no broad panel or submission**  
Locked: 2026-08-05

The sealed result is recorded in
`TARGET80_MAXBELT_MEWTWO_STARMIE_V1_RESULT.md`.

## Hypothesis

The Hancang recovery list improved the frozen policy by 4.46 points in a broad
screen but remained 8-72-1 across the two Starmie families. Its Hero's Cape
cannot improve the two-hit prize race. Replace Cape with Maximum Belt and add a
narrow public-state guard that converts a fully fueled Erasure Ball from 280
to exactly 330 damage against Mega Starmie ex.

The guard attaches Belt to an active Mewtwo only when:

- Mega Starmie ex is Active with 281-330 HP;
- Erasure Ball is legal;
- at least two actual Energy cards are attached to the Rocket Bench; and
- the legal attachment targets that active Mewtwo.

At the ensuing official Erasure Ball prompt, it discards exactly the one or two
Energy cards needed to finish Mega Starmie. It uses no hidden state and does
not alter decisions against any other target. Maximum Belt also remains
available to the frozen neural policy outside the guard condition.

## Frozen identities

Control:

- `work/target80_candidates/hancang_wobbfree_scale_control_v1`;
- deck SHA-256:
  `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`;
- main SHA-256:
  `3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341`.

Candidate:

- `work/target80_candidates/hancang_maxbelt_mewtwo_lethal_v1`;
- deck file SHA-256:
  `6154b27ab5b688a7e7acdf2030c54146228a177fd1defd8ddb58e55cba2b89e7`;
- main SHA-256:
  `6bca147ea05f9a5089a7bb6c9db9d86237eeea203275f66dd19edcac49c236d7`.

Both use model SHA-256
`3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
The deck sequences differ only by Hero's Cape `1159` to Maximum Belt `1158`.
Both are legal 60-card Wobbuffet-free decks.

Collector SHA-256:
`b66a3f5200fdc797cdaeffa6c2ace49f2a03ca64fc4c86e4d23008076a7f50b9`.
Synthetic boundary tests passed, and a four-game side-swapped official smoke
finished 2-2 with zero legality or runtime failures. That smoke is not part of
the locked outcome panel.

## Locked focused panel

Each row interleaves 20 candidate and 20 control games. Do not inspect outcomes
until all four manifests exist.

| Opponent | Rocket order | Seed | Games per arm |
|---|---|---:|---:|
| `SparkStarmieFroslassCFV13_extracted` | first | 806101 | 20 |
| `SparkStarmieFroslassCFV13_extracted` | second | 806102 | 20 |
| `SparkStarmieMistFlatV20_20260801_extracted` | first | 806103 | 20 |
| `SparkStarmieMistFlatV20_20260801_extracted` | second | 806104 | 20 |

Total: 160 genuine official-engine games, 80 per arm, 40 candidate games per
family, and 40 candidate games per start order. Packaged policies run in
isolated workers with strict legality. The collector records exact mechanism
counts in each terminal row.

## Integrity and admission gates

Reject on any changed hash, nonterminal game, illegal action, fallback,
timeout, missing row, wrong schedule count, Wobbuffet, or hidden-information
use. Draws count as non-wins.

Advance only if every condition passes:

1. candidate wins at least 56/80 (`70%`);
2. candidate wins at least eight more games than control (`+10` points);
3. candidate wins at least 24/40 against each Starmie family;
4. candidate wins at least 24/40 from each start order;
5. the exact Belt lethal attachment and discard route completes in at least
   four games; and
6. zero legality or execution failures.

These are only focused-development gates. Passing would authorize a fresh
broad panel, not submission. A later immutable 200-game panel must still reach
160 wins before one internal-arena identity may be submitted. Failure closes
this exact Cape-to-Belt plus Mewtwo lethal hypothesis without threshold mining.
V5 and V6 remain immutable; Kaggle submission is out of scope.
