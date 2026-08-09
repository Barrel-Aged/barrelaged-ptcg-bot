# Target-80 Decidueye/Spidops Bridge V1 result

Status: **closed; failed the frozen development gate**  
Evaluated: 2026-08-04

## Locked result

All three 20-game official-engine arms completed before any outcome was read.
Every game was terminal with no invalid action, exception, timeout, or
step-limit result. The one-game smoke was excluded.

| Arm | Seat 0 | Seat 1 | Total |
|---|---:|---:|---:|
| untouched Wobbuffet V5 | 5-5 | 5-5 | 10-10 |
| closed Wobbuffet-free Recovery V2 | 7-3 | 3-7 | 10-10 |
| Decidueye/Spidops Bridge V1 | 0-10 | 2-8 | 2-18 |

V1 required at least 16/20, at least four more wins than each comparator, and
at least 7/10 from each seat. It failed every strength gate and cannot advance.

## Mechanism audit

- Rowlet appeared in 20/20 games.
- Decidueye appeared in 10/20 games (gate: Rowlet or Decidueye in at least 16,
  but evolution reliability was only 50%).
- Power Shot occurred in 5/20 games, 14 times total (gate: at least 12 games).
- Rocket Rush occurred in 6/20 games, eight times total.
- Either Power Shot or Rocket Rush occurred in only 10/20 games (gate: 18).
- Mewtwo never attacked, satisfying the ceiling of four games.

The payoff conditional on a sustained Decidueye line was strong but rare: the
only two wins used six and four Power Shots respectively, and every game with
at least four Power Shots won. No losing game reached that threshold. Complete
turn review found three repeatable access/sequencing failures rather than a
damage shortfall:

1. Rare Candy could not be searched, leaving Rowlet and Decidueye stranded in
   otherwise adequate hands.
2. Grass was attached to Mimikyu, Spidops, or an already charged Decidueye
   instead of being reserved in hand for Power Shot's mandatory discard.
3. Switch could move an evolved Decidueye out of the Active Spot before it was
   ready to convert an attack.

Those findings may justify a separately frozen consistency package with a
Trainer-search route and explicit attachment/switch conservation. They do not
rescue this exact V1 identity.

## Evidence

- Protocol: `TARGET80_DECIDUEYE_SPIDOPS_BRIDGE_V1_PROTOCOL.md`
- Sealed panel: `work/target80_decidueye_development_v1/`
- Candidate deck SHA-256:
  `4226c1926a4f3c3952bfd632cc3fdd0509d156334c3d36b3c7037e4be2b6a802`
- Candidate main SHA-256:
  `1a6f03c632ce514a03f8af4c81a2f441d7862194cb1ae410dd84df8ca8b5fee7`

This exact package must not be renamed, packaged, submitted, or pooled with
another identity. No Kaggle or internal-arena submission was made.
