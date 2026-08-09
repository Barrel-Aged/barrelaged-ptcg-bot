# Target-80 Golbat / Spidops Bite Single V2 — Locked Confirmation Protocol

Status: preregistered before confirmation outcomes are generated or inspected.

Date: 2026-08-04 (America/Denver)

## Hypothesis

Against the public Marnie/Grimmsnarl archetype, one Zubat–Golbat line can place
20 damage on Grimmsnarl ex with Sneaky Bite. A board containing Golbat plus five
Team Rocket Pokémon then lets weakness-doubled Rocket Rush convert 300 damage
into an exact 320-HP knockout. Carrying only one copy of each bat preserves
three proven Team Rocket support Basics and avoids the two-line bench dilution
seen in exploratory V1.

The Marnie-specific override is gated only by public card IDs. Against every
other archetype the frozen neural policy, early-Spidops Poke Pad guard, and
lethal-only Brave Bangle guard remain the decision path. This is an
official-engine tactical overlay, not a claim that heuristic search is the
project's final game-theoretic architecture.

## Frozen candidate

Directory: `work/target80_candidates/golbat_spidops_bite_single_v2`

- `deck.csv`: `9b20205c7cc4f4d4df67971f100469c25595e50ceeefad7796aee4c9c9db3255`
- `main.py`: `bbdbf72dd569b993d71144ff241c7e69d3e5605816d725f6c28544eeec018961`
- `model.pt`: `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`

Deck integrity requirements:

- exactly 60 cards;
- Wobbuffet (432): 0;
- Zubat (456): 1;
- Golbat (457): 1;
- Team Rocket's Articuno (414): 1;
- Team Rocket's Murkrow (463): 2.

The candidate may not be edited during this confirmation. Any edit creates a
new identity and requires a new protocol.

## Frozen control

Directory: `work/target80_candidates/earlypad_bangle_lethal_v1`

- `deck.csv`: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`
- `main.py`: `80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6`
- `model.pt`: `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`

This is the immutable source of the live EarlyPadLethal V5 submission.

## Official-engine panel

Run `tools/evaluate.py` from the arena lab. Every arm alternates seats and uses
decision traces. Generate all eight arms before reading any confirmation
outcome.

| Opponent | Opponent identity | Candidate games | Control games |
|---|---|---:|---:|
| Marnie | `AshV178MarnieFullBest_extracted` | 24 | 24 |
| Starmie | `SparkStarmieFroslassCFV13_extracted` | 12 | 12 |
| Alakazam | `VioletAlakazamMKV93_extracted` | 12 | 12 |
| Rocket mirror | frozen V5 control | 12 | 12 |

Total: 120 official-engine games, 60 per policy identity.

Opponent hashes:

- Ash Marnie deck/main/model: `92b92bac...`, `0d908014...`, `98072b2a...`.
- Starmie CF13 deck/main: `60e91bf5...`, `3d570642...`.
- Violet Alakazam deck/main: `4e56a202...`, `c324c943...`.

## Admission gates

All gates are conjunctive:

1. All 120 games terminate normally; no draw, timeout, stderr, illegal action,
   or engine exception counts as a pass.
2. Candidate wins at least 34/60 overall and at least four more games than the
   control over its 60 games.
3. Candidate wins at least 13/24 against Ash Marnie and is not worse than the
   control in that arm.
4. Candidate wins at least 4/12 in each non-Marnie arm. Across the 36
   non-Marnie games it may trail control by no more than two wins.
5. Sneaky Bite activates in at least 8/24 Marnie games and the Marnie-specific
   override activates in zero non-Marnie games.
6. Frozen hashes and deck integrity reproduce exactly after the panel.

Passing permits packaging and one immutable internal-arena submission; it does
not prove the persistent 80% objective. Completion still requires that single
submitted identity to record at least 160 wins in its own first 200 rated arena
games, with draws treated as non-wins and no pooling across identities.

Failure means no submission of this identity. Preserve the live V5 bot and
record the exact failed gate before choosing a materially different route.

## Pre-confirmation evidence boundary

- Exploratory two-line V1 went 6-6 against Ash Marnie while the parallel V5
  control went 4-8. Those games are development evidence only and are excluded
  from this confirmation.
- The frozen single-line V2 passed one official-engine smoke: terminal win,
  Sneaky Bite activation and target selection observed, no error. The smoke is
  excluded from the panel.

