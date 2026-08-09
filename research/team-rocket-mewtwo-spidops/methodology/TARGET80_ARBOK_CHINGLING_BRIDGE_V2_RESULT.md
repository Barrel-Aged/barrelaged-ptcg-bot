# Target-80 Arbok / Chingling Bridge V2 result

Status: **closed; failed the frozen development gate**  
Evaluated: 2026-08-04

## Locked result

All three 20-game official-engine arms completed before any outcome was read.
Every game was terminal, both stderr files were empty, and no exception,
invalid action, timeout, or step-limit result occurred. The two earlier smoke
games remain excluded.

| Arm | Seat 0 | Seat 1 | Total |
|---|---:|---:|---:|
| untouched Wobbuffet V5 | 5-5 | 5-5 | 10-10 |
| closed Arbok Lock V1 | 4-6 | 4-6 | 8-12 |
| Wobbuffet-free Arbok/Chingling V2 | 5-5 | 3-7 | 8-12 |

V2 required at least 16/20, at least 7/10 from each seat, four more wins than
V5, and three more wins than Arbok V1. It failed every strength comparison and
cannot advance.

## Mechanism audit

The bridge activated often enough to reject the strategic idea rather than
attribute failure to routing:

- Chiming Commotion occurred in 8/10 second-seat games and 14/20 overall;
- second-seat Chingling averaged 1.8 attacks per game;
- first-seat Arbok was Active before a public Grimmsnarl ex in 8/10 games;
- all 20 games exposed a public Marnie signature; and
- V2 used no Wobbuffet.

Games with any Chiming Commotion went 3-11. Games without it went 5-1. The
zero-Energy hand disruption was therefore not merely too rare; in this frozen
package it consumed tempo and board space without reliably preventing the
Grimmsnarl prize engine. The encouraging one-game second-seat smoke was a
fluke, and Arbok's earlier 10-4 conditional record did not reproduce as a
general 80% route.

This exact V2 deck, setup preference, Switch route, and Chingling postfilter
are closed. Do not rename, rescale, package, submit, or pool them with another
identity.

## Evidence

- protocol: `TARGET80_ARBOK_CHINGLING_BRIDGE_V2_PROTOCOL.md`;
- sealed panel: `work/target80_arbok_chingling_development_v2/`;
- candidate deck SHA-256:
  `448bcb4519dfb0a8be06b3ae5bc49e7888ca8bd207f1ed47cfbf2a014ad7cf65`;
- candidate main SHA-256:
  `7e9a1c2c7bfba00ae9a9e8d2296b1f7e6ab8d9bf4af0284d57b4c799d8fc94eb`.

No Kaggle or internal-arena submission occurred.

