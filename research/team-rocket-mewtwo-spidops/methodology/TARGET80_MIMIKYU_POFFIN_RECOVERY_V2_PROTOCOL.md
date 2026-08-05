# Target-80 Mimikyu Poffin Recovery V2 protocol

Status: **frozen before development outcomes**  
Frozen: 2026-08-04

## Hypothesis

Poffin Spidops V1 improved the exact Marnie cell to 10-10, filled the board in
17/20 games, and reduced Mewtwo-attacking games to two, but it still suffered
attack downtime after the first Spidops line fell. The supplied professional
list uses three Night Stretcher and no Energy Search. V2 therefore makes only
the following deck changes relative to V1:

- Buddy-Buddy Poffin: 3 -> 4;
- Night Stretcher: 1 -> 3;
- Bug Catching Set: 3 -> 2; and
- Energy Search: 2 -> 0.

The Wobbuffet-free Pokemon roster, 11-Energy count, V5 neural model, Poffin
setup behavior, public-Marnie guards, EarlyPad rule, and lethal-Bangle rule are
otherwise byte-identical to V1. This tests an explicit resource-recovery
hypothesis rather than another policy rewrite.

## Frozen identities

| Artifact | SHA-256 |
|---|---|
| untouched V5 deck | `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852` |
| Poffin V1 deck | `4acdc0d75effc2a25fba3feb819d8290a4ca437c4256118530aa61ee9131a9db` |
| Recovery V2 deck | `3f00e230122a1ebc275ebea8ed516ba9cc7f2148ce935417534aa4e9b7602433` |
| shared V1/V2 main | `44296e77d82d7370cb98b92399346d525c26ce9149eb614a6b1d49e9d08f96fb` |
| shared V5 neural model | `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b` |

The exact opponent is the archived `AshV178MarnieFullBest` bundle with
main/deck/model hashes
`0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b`,
`92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d`,
and `98072b2a75197ad9d073af3c7ba5ff66a6daadd266b86b5463fef2ff34910757`.

The terminal one-game V2 smoke in
`work/target80_mimikyu_recovery_smoke_v2/` is excluded.

## Locked development cell

Run fresh, concurrently sealed, automatically side-swapped 20-game arms for:

1. untouched Wobbuffet V5;
2. closed Poffin Spidops V1 as the exact policy comparator; and
3. Poffin Recovery V2.

Use the exact Marnie opponent, `max_steps=300`, `action_timeout_s=10`, and
decision traces. Inspect no outcome until all three arms finish.

V2 advances only if every game is terminal/correct and all gates pass:

- at least 16/20 wins;
- at least four more wins than each contemporaneous comparator;
- at least 7/10 wins from each seat;
- Poffin resolves in at least 17 games;
- Night Stretcher resolves in at least eight games;
- a full board is observed in at least 16 games;
- Spidops attacks in at least 16 games;
- at least two Rocket Rush attacks occur in at least 12 games; and
- Mewtwo attacks in no more than four games.

Failure permanently closes this exact V2 deck. No rename, packaging,
submission, or identity pooling is permitted.

## If admitted

Run the broad-health and untouched-confirmation gates frozen in
`TARGET80_ARBOK_LOCK_V1_PROTOCOL.md`, including both Starmie variants, Rocket,
Alakazam, resource-wall opponents, and both seats. Only one independently
confirmed package may be submitted to the internal arena.

Success remains at least 160 wins in one immutable successor's own first 200
rated games; draws are non-wins. No Kaggle submission is authorized. Local
official-engine evidence is not arena proof.
