# Target-80 Mimikyu Poffin Spidops V1 result

Status: **closed; failed the frozen development gate**  
Evaluated: 2026-08-04

## Locked result

All three 20-game arms ran concurrently against the exact archived
`AshV178MarnieFullBest` package and remained sealed until every arm had
finished. The one-game smoke was excluded.

| Arm | Seat 0 | Seat 1 | Total | Terminal |
|---|---:|---:|---:|---:|
| untouched Wobbuffet V5 | 4-6 | 4-6 | 8-12 | 20/20 |
| Wobbuffet-free Mimikyu control | 6-4 | 3-7 | 9-11 | 20/20 |
| Mimikyu Poffin Spidops V1 | 5-5 | 5-5 | 10-10 | 20/20 |

The candidate was two wins better than V5 and one better than the same-deck
control, but required at least 16/20 and a four-win advantage over both. It
also required at least 7/10 from each seat. It failed all strength gates and
cannot advance.

## Mechanism audit

The intended mechanism was real and passed every behavioral gate:

- Poffin resolved in 17/20 games (gate: at least 10);
- a full six-Pokemon board was observed in 17/20 (gate: at least 16);
- Spidops attacked in 17/20 (gate: at least 16);
- Mewtwo attacked in 2/20 (gate: no more than four);
- Hero's Cape reached a Tarountula/Spidops line in 12 games; and
- all games ended normally with no legality or agent failure.

All ten wins filled the board and attacked with Spidops; none attacked with
Mewtwo. The winning games launched their first Rocket Rush on turn 4.7 on
average. The losses averaged turn 13.8 when never-attacking games are assigned
the frozen turn-30 sentinel. Thus Poffin and the bounded guards repaired much
of the intended setup behavior, but not enough games launched the pressure
chain early enough to approach the target.

This exact 11-Energy/three-Poffin deck and guard package is permanently closed.
It must not be renamed, packaged, submitted, or pooled with another identity.

## Evidence

- Protocol: `TARGET80_MIMIKYU_POFFIN_SPIDOPS_V1_PROTOCOL.md`
- Sealed panel: `work/target80_mimikyu_poffin_development_v1/`
- Candidate deck SHA-256:
  `4acdc0d75effc2a25fba3feb819d8290a4ca437c4256118530aa61ee9131a9db`
- Candidate main SHA-256:
  `44296e77d82d7370cb98b92399346d525c26ce9149eb614a6b1d49e9d08f96fb`

No Kaggle or internal-arena submission was made from this result.
