# Target-80 Decidueye/Spidops Bridge V1 protocol

Status: **frozen before development outcomes**  
Frozen: 2026-08-04

## Hypothesis

Recovery V2 made Spidops attack in all 20 exact-Marnie games and still went
10-10. Iron/Arbok exploration then showed why ability-bearing walls are not a
complete answer: Froslass and Munkidori can win through damage counters without
depending on attack damage.

Decidueye is a one-prize, no-Ability Grass Pokemon. Power Shot costs one Grass,
does 170 damage, and discards a Basic Grass Energy from hand. It therefore:

- is not a target for Froslass's Freezing Shroud;
- reaches 340 damage through Marnie's Grimmsnarl ex's Grass Weakness, enough
  to take two prizes in one attack;
- trades one prize for two if Grimmsnarl attacks it back; and
- puts Grass Energy into the discard pile for Spidops's Charging Up.

V1 is a Wobbuffet-free package redesign with three Rowlet, three Decidueye,
four Rare Candy, four Hilda, four Buddy-Buddy Poffin, four Poke Pad, three
Ultra Ball, three Switch, two Energy Retrieval, two Night Stretcher, and Prime
Catcher. It retains four Tarountula, four Spidops, two Mewtwo ex, two Mimikyu,
seven Grass, one Psychic, and four Team Rocket's Energy. A public-state router
prioritizes the Decidueye line only after a Marnie-family card is visible; the
same V5 neural policy remains the fallback.

## Frozen identities

| Artifact | SHA-256 |
|---|---|
| untouched V5 deck | `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852` |
| untouched V5 main | `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6` |
| Recovery V2 deck | `3f00e230122a1ebc275ebea8ed516ba9cc7f2148ce935417534aa4e9b7602433` |
| Recovery V2 main | `44296e77d82d7370cb98b92399346d525c26ce9149eb614a6b1d49e9d08f96fb` |
| Decidueye V1 deck | `4226c1926a4f3c3952bfd632cc3fdd0509d156334c3d36b3c7037e4be2b6a802` |
| Decidueye V1 main | `1a6f03c632ce514a03f8af4c81a2f441d7862194cb1ae410dd84df8ca8b5fee7` |
| shared V5 neural model | `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b` |

The exact opponent is the archived `AshV178MarnieFullBest` bundle with
main/deck/model hashes
`0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b`,
`92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d`,
and `98072b2a75197ad9d073af3c7ba5ff66a6daadd266b86b5463fef2ff34910757`.

The terminal one-game smoke in `work/target80_decidueye_smoke_v1/` is
excluded. It demonstrated the intended sequence (Rowlet, Ultra Ball,
attachment, Switch, Rare Candy, Power Shot) but is not selection evidence.

## Locked development cell

Run fresh, concurrently sealed, automatically side-swapped 20-game arms for:

1. untouched Wobbuffet V5;
2. closed Wobbuffet-free Recovery V2; and
3. Decidueye/Spidops Bridge V1.

Use the exact Marnie opponent, `max_steps=300`, `action_timeout_s=10`, and
decision traces. Inspect no outcome until all three arms finish.

V1 advances only if every game is terminal/correct and all gates pass:

- at least 16/20 wins;
- at least four more wins than each contemporaneous comparator;
- at least 7/10 wins from each seat;
- Rowlet or Decidueye appears in at least 16 games;
- Decidueye uses Power Shot in at least 12 games;
- at least one Power Shot or Rocket Rush occurs in at least 18 games; and
- Mewtwo attacks in no more than four games.

Failure permanently closes this exact deck and policy. No rename, packaging,
submission, or identity pooling is permitted.

## Downstream gates if admitted

Run the broad-health and untouched-confirmation gates frozen in
`TARGET80_ARBOK_LOCK_V1_PROTOCOL.md`: a 20-game five-opponent panel requiring
at least 14/20 and no zero-win cell, followed by an untouched 60-game panel
requiring at least 48/60 overall, 32/40 exact Marnie, 14/20 broad, and all
correctness gates. Only then run compile, deck-integrity, official smoke,
self-contained packaging, archive-member, and packaged-runtime checks.

Only one immutable successor may be submitted to the internal arena. Goal
completion remains at least 160 wins in that identity's own first 200 rated
games; draws are non-wins and identities may not be pooled. No Kaggle
submission is authorized. Local official-engine results prove controlled
execution, not arena strength.
