# Target-80 Mimikyu Poffin Recovery V2 result

Status: **closed; failed the frozen development gate**  
Evaluated: 2026-08-04

## Locked result

All three 20-game arms ran concurrently against the exact archived
`AshV178MarnieFullBest` package and remained sealed until every arm had
finished. The separate one-game smoke was excluded.

| Arm | Seat 0 | Seat 1 | Total | Terminal |
|---|---:|---:|---:|---:|
| untouched Wobbuffet V5 | 7-3 | 5-5 | 12-8 | 20/20 |
| closed Poffin Spidops V1 | 5-5 | 4-6 | 9-11 | 20/20 |
| Mimikyu Poffin Recovery V2 | 7-3 | 3-7 | 10-10 | 20/20 |

Recovery V2 required at least 16/20 wins, at least four more wins than each
contemporaneous comparator, and at least 7/10 wins from each seat. It failed
all three strength conditions and cannot advance.

## Mechanism audit

The recovery mechanism fired extensively and eliminated the earlier Spidops
availability symptom:

- Buddy-Buddy Poffin resolved in 15/20 games (gate: at least 17);
- Night Stretcher resolved 28 times across 18/20 games (gate: at least eight
  games);
- a full six-Pokemon board was observed in 18/20 games (gate: at least 16);
- Spidops attacked in 20/20 games (gate: at least 16);
- at least two Rocket Rush attacks occurred in 16/20 games (gate: at least
  12); and
- Mewtwo attacked in 1/20 games (gate: no more than four).

The candidate passed every behavioral gate except Poffin frequency. More
importantly, wins and losses both launched their first Rocket Rush on turn 7.3
on average. Once setup and recovery were made reliable, neither metric
separated wins from losses. The remaining exact-Marnie deficit is therefore
not credibly addressed by further Poffin, Night Stretcher, Energy Search, or
ordinary setup-count tuning. It requires a distinct denial, prize-map, or
tactical mechanism.

This exact Recovery V2 deck is permanently closed. It must not be renamed,
packaged, submitted, or pooled with another identity.

## Evidence

- Protocol: `TARGET80_MIMIKYU_POFFIN_RECOVERY_V2_PROTOCOL.md`
- Sealed panel: `work/target80_mimikyu_recovery_development_v2/`
- Candidate deck SHA-256:
  `3f00e230122a1ebc275ebea8ed516ba9cc7f2148ce935417534aa4e9b7602433`
- Candidate main SHA-256:
  `44296e77d82d7370cb98b92399346d525c26ce9149eb614a6b1d49e9d08f96fb`
- Shared V5 neural model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`

No Kaggle or internal-arena submission was made from this result.
