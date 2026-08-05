# Target-80 Cornerstone Wall V1 result

Status: **closed; failed the frozen development gate**  
Evaluated: 2026-08-04

## Locked result

Both 20-game arms were run concurrently against the exact archived
`AshV178MarnieFullBest` bundle and were not inspected until both processes had
finished. The earlier one-game smoke remained excluded.

| Arm | Seat 0 | Seat 1 | Total | Terminal |
|---|---:|---:|---:|---:|
| untouched V5 control | 7-3 | 4-6 | 11-9 | 20/20 |
| Cornerstone Wall V1 | 3-7 | 1-9 | 4-16 | 20/20 |

The candidate missed every strength threshold: it required at least 16/20,
at least four more wins than control, and at least 7/10 from each seat. It was
seven wins worse than the contemporaneous control.

## Mechanism audit

The mechanism did activate, which makes the negative result diagnostic rather
than an inconclusive setup miss:

- Cornerstone Mask Ogerpon ex reached the Active Spot in 19/20 games (gate:
  at least 12).
- It used Demolish 62 times across 16 games.
- Hero's Cape was observed on an Active Cornerstone in 12 games.
- Official-engine logs recorded 234 damage-counter placement events on
  Cornerstone in the candidate's own decision stream.

Cornerstone Stance correctly prevents attack damage from Pokemon with
Abilities, but it does not prevent Froslass and Munkidori damage-counter
effects. The resulting 140-damage, three-energy attacker was too slow to win
the prize race and became a two-prize liability. This exact deck and public
pilot are permanently closed; they must not be packaged, submitted, or
retested under another name.

## Evidence

- Protocol: `TARGET80_CORNERSTONE_WALL_V1_PROTOCOL.md`
- Sealed panel: `work/target80_cornerstone_development_v1/`
- Candidate deck SHA-256:
  `61ef3d9b28cbdf9a477235b4aebd202b54d9b3296a163160f0210edcf0793ff3`
- Candidate main SHA-256:
  `fc11043d88020d2544dc96406a91a8b15e9bbe8ce92a5afde4f6dea7e938cd71`

No Kaggle or internal-arena submission was made from this result.
