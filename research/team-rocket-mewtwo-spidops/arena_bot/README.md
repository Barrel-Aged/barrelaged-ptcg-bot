# Final arena bot source snapshot

`main.py` is the exact authored entrypoint submitted as
`HelbrosRocketSpidopsArticunoV7`; its SHA-256 is
`d690d9f316180ebd87f5648257deacbaaf8a9bb7bd50a5231ed8b128f6be2484`.

The entrypoint combines a frozen greedy neural action policy with three narrow
deterministic guards:

1. Early Poke Pad searches for a missing Spidops when an in-play Tarountula has
   no matching evolution in hand.
2. Brave Bangle is forced only when it changes a legal Spidops Rocket Rush from
   nonlethal to an ex knockout.
3. At four occupied bench slots, a publicly revealed attack-effect matchup
   causes Articuno to be played if legal. If Articuno is not yet in hand, only
   a duplicate Mewtwo ex, duplicate Mimikyu, or fourth Tarountula is masked.

The Articuno guard does not trigger on damage-only attacks: Repelling Veil
prevents effects of attacks done to Basic Team Rocket Pokemon, while damage is
explicitly not an effect.

## Runtime boundary

The official engine, packaged `soldier` inference runtime, model checkpoint,
and opponent-prior library are excluded from this repository. Consequently,
this directory is an auditable source snapshot, not a standalone arena bundle.
See the repository-level `PUBLICATION_BOUNDARY.md`.

## Exact final deck

- 8 Basic Grass Energy
- 4 Team Rocket's Energy
- 4 Team Rocket's Tarountula
- 4 Team Rocket's Spidops
- 2 Team Rocket's Articuno
- 2 Team Rocket's Mewtwo ex
- 2 Team Rocket's Mimikyu
- 1 Buddy-Buddy Poffin
- 3 Bug Catching Set
- 2 Night Stretcher
- 1 Ultra Ball
- 4 Team Rocket's Transceiver
- 4 Poke Pad
- 1 Hero's Cape
- 2 Brave Bangle
- 4 Team Rocket's Ariana
- 2 Team Rocket's Giovanni
- 1 Team Rocket's Petrel
- 2 Team Rocket's Proton
- 4 Lillie's Determination
- 3 Team Rocket's Factory

Total: 60 cards. Team Rocket's Wobbuffet: 0.

