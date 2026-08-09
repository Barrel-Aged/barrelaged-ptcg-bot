# Target-80 Pikachu ex / Crispin / Sparkling Crystal V1R smoke result

Status: **mechanism verified; exact route closed at health gate**  
Recorded: 2026-08-05 (America/Denver)

## Hypothesis

Mega Starmie ex is Lightning-weak and gives three Prizes. Pikachu ex (`210`)
can use Topaz Bolt (`282`) for 300 base damage, while Sparkling Crystal
(`1165`) reduces the attack's Grass / Lightning / Metal cost by one Energy.
With Crystal removing the Metal requirement, Crispin can establish one of the
remaining Energy types and a manual attachment can establish the other. The
intended line therefore converts two Energy into a weakness-amplified knockout
on Mega Starmie.

This deterministic public-state wrapper was a mechanism and trajectory-health
probe. It was never eligible as the final strategic policy.

## Frozen debug identity

Candidate:
`work/target80_candidates/pikachu_crispin_crystal_route_v1r`

- 60 cards and zero Wobbuffet;
- 2 Pikachu ex, 4 Crispin, 2 Team Rocket's Petrel, 4 Team Rocket's
  Transceiver, 1 Sparkling Crystal, 4 Grass Energy, 4 Lightning Energy, and
  4 Team Rocket's Energy;
- deck SHA-256:
  `7754f6549cc58f398904fb7451cee28c3ba4b2b98dff084489aec023af5798d5`;
- model SHA-256:
  `b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`;
- corrected main SHA-256:
  `373baa0186e537cdb2c149e5a480230029bb97a3682f8b349a1a84f78b7e3d0f`.

The V1R correction moved manual typed-Energy attachment ahead of optional
board search after Topaz Bolt discarded its attached Energy and prevented an
active Crystal Pikachu from being retreated by the fallback policy.

## Open smoke result

The five V1R games against `StarmieMist` were implementation-visible health
checks, not a preregistered or sealed strength panel:

| Artifact | Record |
|---|---:|
| `smoke1.jsonl` | 0-1 |
| `smoke4.jsonl` | 1-3 |
| Combined | **1-4** |

The intended official-engine mechanism did execute:

- Pikachu appeared and became Active in four of five games;
- Sparkling Crystal reached Pikachu in three of five;
- Pikachu became Grass-plus-Lightning ready in two of five;
- Topaz Bolt was used three times across those two games; and
- all three attacks targeted a full-health Mega Starmie.

The two games where the mechanism fired went 1-1. The win required two Topaz
Bolt attacks; the loss converted only one. The other three games never
produced a ready attacker. Crystal acquisition and the repeat-attacker /
post-discard recharge chain therefore remained too unreliable to justify a
sealed panel.

## Decision

Close this exact V1/V1R deck/model/wrapper at the smoke-health gate. Do not
train on its wrapper choices, package it, submit it, or treat the 1-4 sample as
a formal strength estimate. A future Pikachu route would need a materially
different deck-level engine that solves Crystal access, recharge, backup
attacker continuity, and broad-matchup prize liability before it is tested.

The next Target-80 route is the queued exact-deck learning path on a coherent
Wobbuffet-free substrate, not another post-hoc edit to this wrapper.

## Evidence boundary

The games, legal actions, transitions, attacks, damage, Prizes, and outcomes
were genuine official-engine execution. The named route and health criteria
are research abstractions. This smoke does not establish public-belief search,
safe re-solving, exploitability, internal-arena strength, or an 80% win rate.

Evidence:

- `work/target80_pikachu_crystal_smoke_v1r/smoke1.trace.jsonl`, SHA-256
  `72eaf9a6c4f67e85e9c733286023538019b08a373d7c7c2ba112315d898c719c`;
- `work/target80_pikachu_crystal_smoke_v1r/smoke4.trace.jsonl`, SHA-256
  `aaa612b39f17be64a186bbdf8cda622124b51e4b5b2f26789017a62b43bdd4ae`;
- identity manifest:
  `work/target80_candidates/pikachu_crispin_crystal_route_v1r/PIKACHU_CRYSTAL_V1_PROVENANCE.json`.
