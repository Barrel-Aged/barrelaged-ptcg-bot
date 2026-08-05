# Target-80 Miraidon/Crispin conversion V1 protocol

Status: **completed and rejected at Stage A; no deployment authority**

Preregistered: 2026-08-05 (America/Denver)

## Why this is a different hypothesis

Mega Starmie ex's `Nebula Beam` does 210 damage and explicitly ignores effects
on the opposing Active Pokemon. That makes active damage-prevention walls a
poor structural answer. Mega Starmie is Lightning-weak.

Miraidon ex (engine card 957) is a Basic Lightning Pokemon with 220 HP. Its
`Hadron Spark` (attack 1377) does 120 plus 120 when the opposing Active is a
Pokemon ex. Against Lightning-weak Staryu or Mega Starmie, the official card
math is therefore 240 or 480 damage respectively; Miraidon also survives one
210-damage Nebula Beam at 10 HP. Crispin can search two different Basic Energy
types, attach one, and put the other into hand, creating a coherent two-turn
route to Miraidon's two-Lightning-plus-Colorless cost.

This is not another wall, Galvantula derivative, global elite-BC update, or
post-hoc action-label rule. It tests a Basic attacker with its own energy and
search package. The public-state route is only a mechanism probe. Even if it
passes, the deterministic bridge cannot be the final strategy; success would
authorize exact-deck paired supervision and learned policy work before full
match confirmation.

## Frozen common policy

All three arms use the completed but development-rejected V1R iteration-3
checkpoint as a common research seed:

`b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`

It is not represented as an admitted or deployable policy. Its role here is
only to hold the neural decision substrate constant while testing the deck and
route.

### Hancang learned control

`work/target80_candidates/hancang_elitebc_iter3_control_v1`

- semantic tree SHA-256:
  `abbe2a41d279c3227d4d6e7436ce89745608e025ee597e2283f6cbe2e270a24b`
- deck SHA-256:
  `0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8`
- main SHA-256:
  `3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341`

### Miraidon deck control

`work/target80_candidates/miraidon_crispin_deck_control_v1`

- semantic tree SHA-256:
  `469a27d635b0cc7bf509081f04e882e9d61e941e82e5391daef0f8dbcb0e31f5`
- deck SHA-256:
  `1b1fb0a8f1adbbae3324d9d3c16cb1324035aa6b82d5576f7b1141b39f9f8040`
- unchanged neural-control main SHA-256:
  `3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341`

### Miraidon public-Starmie route

`work/target80_candidates/miraidon_crispin_route_v1`

- semantic tree SHA-256:
  `5f808f25bc0839bd2652846e6a7c57de04e09da5523011b258441bb67eccada5`
- same candidate deck SHA-256:
  `1b1fb0a8f1adbbae3324d9d3c16cb1324035aa6b82d5576f7b1141b39f9f8040`
- routed main SHA-256:
  `79e69e6eaa8ffb1423ea9dd52523673358948c41bce68b13629d556a3d3df236`

The runtime handshake and synthetic public-state tests passed for Crispin play
and search, manual Lightning attachment, Hadron attack selection, 60-card deck
integrity, and zero Wobbuffet. No official match outcome was generated or
inspected before this lock.

## Exact candidate deck (60 cards)

- Energy: 6 Basic Grass; 3 Basic Lightning; 4 Team Rocket's Energy.
- Pokemon: 4 Team Rocket's Tarountula; 4 Team Rocket's Spidops; 1 Team
  Rocket's Articuno; 2 Team Rocket's Mewtwo ex; 1 Team Rocket's Mimikyu; 2
  Miraidon ex.
- Trainers: 1 Buddy-Buddy Poffin; 2 Bug Catching Set; 2 Night Stretcher; 4
  Ultra Ball; 4 Team Rocket's Transceiver; 3 Poke Pad; 1 Hero's Cape; 2 Brave
  Bangle; 3 Crispin; 2 Team Rocket's Ariana; 1 Team Rocket's Giovanni; 1 Team
  Rocket's Petrel; 2 Team Rocket's Proton; 3 Lillie's Determination; 2 Team
  Rocket's Factory.

The deck contains zero Wobbuffet and exactly one ACE SPEC. Relative to the
Hancang control it removes one Articuno, one Mimikyu, two Grass, one Bug Set,
one Poke Pad, two Ariana, one Giovanni, one Lillie, and one Factory; it adds
two Miraidon, three Lightning, three Crispin, and three additional Ultra Ball.

## Frozen execution boundary

Opponents:

- `SparkStarmieMistFlatV20_20260801_extracted`
  - semantic tree SHA-256:
    `25690bea08bc73dcd93207b1e3ed77428ed09f1a15d6137251f2e06b6bd7dd79`
  - deck SHA-256:
    `64ef6e2c0f5fadc12f96426aa80a07e1402501ed00d1778a0badd25d171e4901`
- `SparkStarmieFroslassCFV13_extracted`
  - semantic tree SHA-256:
    `c693adec6062422241f2b52caf6beba6518913c2c1ed66581b369eba15b92f76`
  - deck SHA-256:
    `60e91bf519988fd2efd14b8c9806bf03febf8baa2caab5250eab1da65b674d61`

Collector:
`scripts/collect_target80_deck_matched_bc_v1_panel.py`

- SHA-256:
  `8e675a5deb1199bd1f2c4bd9d9b128f2143ae76e36e0f7efc9547cc3149df38f`
- genuine official `cg` engine;
- packaged runtimes in isolated official-evaluator workers;
- strict action legality and no hidden-hand/visualizer access;
- Rocket occupies seat 0 and starts first in every pilot game;
- each output and sidecar must not exist before collection;
- neither opponent file in a stage may be opened or summarized until both
  collectors in that stage finish.

## Stage A: does the public route execute and add value?

Compare the routed candidate with the deck-identical neural control, 12 games
per arm against each opponent (48 total), randomly interleaved:

- MistFlat seed `806301`, output
  `work/target80_miraidon_crispin_v1/mist_route_vs_deck.jsonl`;
- Froslass seed `806302`, output
  `work/target80_miraidon_crispin_v1/fros_route_vs_deck.jsonl`.

Every gate is required:

1. 48 terminal games, exact arm/opponent cells, byte-stable identities, and
   zero engine/worker/legality/fallback errors.
2. Routed candidate at least 18 wins in 24 games (75%).
3. Routed candidate at least six more wins than the deck-identical control.
4. Routed candidate at least eight wins in each 12-game Starmie family.
5. Miraidon appears in at least 18 routed games, becomes ready in at least 14,
   and uses Hadron Spark at least 18 times across at least 12 games.
6. At least 12 Hadron Spark attacks directly target Staryu or Mega Starmie.

If any Stage A gate fails, stop and close the exact deck/route. Do not run
Stage B or rescue a favorable opponent slice.

## Stage B: did the structural package beat the prior shell?

Run only if Stage A passes. Compare the routed candidate with the Hancang
learned control, again 12 games per arm per opponent:

- MistFlat seed `806303`, output
  `work/target80_miraidon_crispin_v1/mist_route_vs_hancang.jsonl`;
- Froslass seed `806304`, output
  `work/target80_miraidon_crispin_v1/fros_route_vs_hancang.jsonl`.

Required gates:

1. All integrity conditions from Stage A.
2. Routed candidate at least 18 wins in 24 games.
3. At least five more wins than Hancang control.
4. At least eight wins in each opponent family.
5. The complete mechanism thresholds from Stage A reproduce.

## Authority after a pass

Passing both stages authorizes only a new exact-deck learning protocol. That
successor must use paired public-state action or plan values, retain a safe
baseline, and replace the mechanism probe with a learned policy before broad
testing. It must then pass side-swapped Marnie, Rocket, Alakazam, Crustle,
Archaludon/Cinderace, and generalist regressions plus independent confirmation.

No result here authorizes packaging, internal-arena submission, Kaggle
submission, or an 80% strength claim. Project completion still requires one
immutable bot to win at least 160 of its own first 200 rated internal-arena
games.

## Result

Both Stage A collectors completed with all 48 terminal rows, exact locked
bundle identities, first-seat convention, and zero engine, worker, legality,
fallback, or hidden-information failures.

| Opponent | Routed candidate | Deck-identical control |
|---|---:|---:|
| MistFlat Starmie | 1-11 | 3-9 |
| Froslass Starmie | 1-10-1 | 1-11 |
| Total | 2-21-1 | 4-20 |

Miraidon appeared in 21/24 routed games and Crispin was played 21 times, but
Miraidon was never observed ready and never used Hadron Spark. The route made
12 manual Lightning-to-Miraidon attachments and selected Miraidon from Ultra
Ball 21 times, yet the full two-Lightning-plus-Colorless conversion did not
occur before the Starmie clock ended the games. Crispin-to-target telemetry was
observed only once, reinforcing that the supporter prompt/attachment sequence
did not produce the intended acceleration.

The routed arm missed the 18-win gate by 16 wins, trailed rather than beat its
deck-identical control, missed both opponent floors, and missed every readiness
and Hadron mechanism floor. Stage A therefore rejects the exact deck/route.
Stage B was not run and is not authorized post hoc.

Artifacts:

- `work/target80_miraidon_crispin_v1/mist_route_vs_deck.jsonl`
  SHA-256 `0a86a0911799f1ff9a94941d18d9982da5b19ec936c45b47d103941fbb85fe43`
- `work/target80_miraidon_crispin_v1/fros_route_vs_deck.jsonl`
  SHA-256 `14cde53cf66a6ebd5843b902afa5d01e448a6c5db36e05eddf3db952288b35a0`

No learned successor, broad panel, package, arena submission, or Kaggle
submission was produced.
