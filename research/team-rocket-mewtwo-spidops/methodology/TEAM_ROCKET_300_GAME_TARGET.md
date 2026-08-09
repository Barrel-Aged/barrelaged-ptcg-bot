# Team Rocket 300-game rank-1 target

Date opened: 2026-07-28  
Status: active

## Completion contract

One immutable submitted bot identity must simultaneously satisfy all of the
following:

1. Use the qualifying 60-card Team Rocket Spidops / Mewtwo deck below.
2. Have at least 300 completed rated internal-arena games.
3. Be current rank 1 after game 300 or later.
4. Remain active on the ladder.
5. Pass an exact-archive audit covering bytes, safe members, deck identity,
   packaged runtime, official-engine legality, game count, and live rank.

The earlier rank-1 Marnie's Grimmsnarl V4 result does not qualify because it is
not a Team Rocket Pokemon deck.

## Frozen qualifying deck

This legal list is shared by the three strongest recovered Team Rocket policy
packages (`SparkRocketSpidopsV1`, `HelbrosRocketSpidopsProV3`, and
`SparkRocketSpidopsStageRLV4`). Deck SHA-256:
`256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

Pokemon (15):

- 4 Team Rocket's Tarountula
- 4 Team Rocket's Spidops
- 2 Team Rocket's Mewtwo ex
- 2 Team Rocket's Articuno
- 2 Team Rocket's Murkrow
- 1 Team Rocket's Wobbuffet

Trainers (31):

- 4 Team Rocket's Ariana
- 4 Team Rocket's Transceiver
- 4 Poke Pad
- 3 Bug Catching Set
- 3 Team Rocket's Giovanni
- 2 Team Rocket's Proton
- 2 Energy Search
- 2 Team Rocket Factory
- 2 Lillie's Determination
- 1 Team Rocket's Petrel
- 1 Team Rocket's Archer
- 1 Night Stretcher
- 1 Hero's Cape
- 1 Brave Bangle

Energy (14):

- 7 Basic Grass Energy
- 3 Basic Psychic Energy
- 4 Team Rocket Energy

## Frozen starting baselines

At target initialization, the strongest existing qualifying policies were:

| Bot | Record | Rating | Rank |
|---|---:|---:|---:|
| SparkRocketSpidopsV1 | 971-763-2 | 877.7 | 37 |
| HelbrosRocketSpidopsProV3 | 945-785-2 | 862.9 | 44 |
| SparkRocketSpidopsGeneralistV1 | 11500-12368-94 | 854.5 | 49 |

These establish that merely replaying an existing Team Rocket package cannot
meet the rank-1 requirement; the policy must improve materially.
