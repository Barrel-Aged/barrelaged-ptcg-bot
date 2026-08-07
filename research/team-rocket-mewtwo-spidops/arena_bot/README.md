# Final arena bot source snapshot

`main.py` is the exact authored entrypoint submitted as
`HelbrosRocketSpidopsBC2RecoveryV8`; its SHA-256 is
`e9c2b0c59116845e6c11e520e8d47ca2f1476a32a69e6dad9fcd8da371dd4c9e`.

The entrypoint combines the exact frozen ScaleBCV2 neural action policy with
four narrow public-state guards:

1. Early Poke Pad searches for a missing Spidops when an in-play Tarountula has
   no matching evolution in hand.
2. Brave Bangle is forced only when it changes a legal Spidops Rocket Rush from
   nonlethal to an ex knockout.
3. At four occupied bench slots, a publicly revealed attack-effect matchup
   causes Articuno to be played if legal. If Articuno is not yet in hand, only
   lower-priority or redundant Basic plays are masked, including the lone
   Wobbuffet.
4. A selected Wobbuffet Headbutt Bounce is redirected only to an offered
   Retreat when a Benched Spidops is already ready for Rocket Rush. Rocket
   Mirror is never intercepted.

The Articuno guard does not trigger on damage-only attacks: Repelling Veil
prevents effects of attacks done to Basic Team Rocket Pokemon, while damage is
explicitly not an effect. The Wobbuffet exit rule changed the two audited bad
Kaggle Headbutt decisions but activated zero times in the final fresh screen,
so its causal win value is not claimed.

## Runtime boundary

The official engine, packaged `soldier` inference runtime, model checkpoint,
and opponent-prior library are excluded from this repository. Consequently,
this directory is an auditable source snapshot, not a standalone arena bundle.
See the repository-level `PUBLICATION_BOUNDARY.md`.

## Exact final deck

- 7 Basic Grass Energy
- 3 Basic Psychic Energy
- 4 Team Rocket's Energy
- 4 Team Rocket's Tarountula
- 4 Team Rocket's Spidops
- 2 Team Rocket's Articuno
- 2 Team Rocket's Mewtwo ex
- 2 Team Rocket's Murkrow
- 1 Team Rocket's Wobbuffet
- 3 Bug Catching Set
- 1 Night Stretcher
- 2 Energy Search
- 4 Team Rocket's Transceiver
- 4 Poke Pad
- 1 Hero's Cape
- 1 Brave Bangle
- 4 Team Rocket's Ariana
- 1 Team Rocket's Archer
- 3 Team Rocket's Giovanni
- 1 Team Rocket's Petrel
- 2 Team Rocket's Proton
- 2 Lillie's Determination
- 2 Team Rocket's Factory

Total: 60 cards. Deck SHA-256:
`256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
