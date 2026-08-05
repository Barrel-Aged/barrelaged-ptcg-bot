# Target-80 Kangaskhan backup V1 result

Recorded: 2026-08-05

Status: **rejected; exact deck-only substitution closed**

## Frozen result

All 24 official-engine games were terminal, with zero draws, crashes, illegal
selections, or timeouts.

| Arm | W-L vs `AshV178MarnieFullBest` |
|---|---:|
| `KangaskhanBackupV1` | 3-9 |
| immutable V5 | 9-3 |

The candidate required at least 9-3 and a two-win lead over V5. It instead
trailed V5 by six wins.

## Mechanism audit

- Kangaskhan reached play in three of twelve games, below the four-game gate.
- It received two Energy attachments in one game but used neither `Comet
  Punch` nor `Wicked Impact` anywhere in the panel.
- All three games in which Kangaskhan reached play were losses.
- The candidate's three wins occurred without Kangaskhan entering play.

The card therefore did not function as the intended ready backup attacker
under the unchanged policy. Its theoretical two-attachment `Wicked Impact`
curve is legal, but theoretical attack math is not evidence that this exact
package can assemble or sequence the line.

## Decision

Close the exact Wobbuffet-to-Kangaskhan deck-only substitution. Do not broaden,
package, submit, or add post-hoc Kangaskhan routing based on these seeds. A
future Kangaskhan hypothesis would need a materially redesigned acceleration
and selection package with a new prospective boundary.

The live V5 package remains unchanged.

## Durable evidence

- protocol: `TARGET80_KANGASKHAN_BACKUP_V1_PROTOCOL.md`;
- candidate: `work/target80_candidates/kangaskhan_backup_v1/`;
- smoke: `work/target80_kangaskhan_backup_smoke_v1.jsonl`;
- frozen panel and traces:
  `work/target80_kangaskhan_backup_explore_v1/`.

Evidence boundary: all legality, card-use, and match statements above come
from official-engine execution. The small frozen screen rejects this package
under its prospective gate; it is not a universal Kangaskhan estimate or a
claim about final neural/game-theoretic planning.
