# Target-80 Galvantula Grim-breaker V1 result

Recorded: 2026-08-04

Status: **rejected; no broad panel, package, or submission authorized**

## Frozen screen

All 36 official-engine games terminated normally with zero draws, crashes,
illegal selections, or timeouts.

| Arm | Deck | W-L vs `AshV178MarnieFullBest` |
|---|---|---:|
| `GalvantulaGrimBreakerV1` | frozen 2-2 Galvantula package | 2-10 |
| `GalvantulaGrimControlV1` | identical 60 cards, original V5 policy | 3-9 |
| immutable V5 | original V5 deck and policy | 3-9 |

The candidate required at least 9-3 and a two-win advantage over each control.
It missed all three outcome gates and was one win worse than both controls.

## Mechanism audit

The tactical premise was real.  Deduplicating the adjacent trace copies gives
17 `Shocking Web` attacks in ten games and ten exact 360-damage knockouts of
Marnie's Grimmsnarl ex.  The route therefore exceeded its required mechanism
floor by a wide margin.  It also established Galvantula in ten of twelve games
and at least Joltik in all twelve.

That success did not convert into match wins:

- Eight of the ten games with a Grimmsnarl knockout were still losses.
- The only two wins were the games where Galvantula survived long enough to
  attack three times, including additional knockouts of the follow-up line.
- Most games produced one favorable one-prize-for-two-prize exchange and then
  lost the remaining race after Marnie rebuilt.
- The 2-2 line, Lightning package, search bridge, and bench commitment reduced
  the ordinary Rocket setup enough that a single counter knockout was not a
  sufficient strategic plan.

## Decision

Close this exact 2-2 Joltik/Galvantula package and its public-Marnie routing
logic.  Do not scale the line, run the broad panel, package it, or submit it.
The durable lesson is sharper than a damage check: a counter attacker must
either remain online for multiple prizes or preserve the Spidops/Mewtwo board
while trading.  One isolated 360-damage knockout is not enough against this
Marnie recovery engine.

The immutable V5 live bot remains unchanged.

## Durable evidence

- preregistration: `TARGET80_GALVANTULA_GRIM_BREAKER_V1_PROTOCOL.md`;
- candidate: `work/target80_candidates/galvantula_grim_breaker_v1/`;
- deck-identical control:
  `work/target80_candidates/galvantula_grim_control_v1/`;
- final smoke:
  `work/target80_galvantula_grim_breaker_smoke_v1b.jsonl`;
- frozen three-arm screen and decision traces:
  `work/target80_galvantula_grim_breaker_explore_v1/`.

Evidence boundary: these are genuine official-engine executions.  The small
development panel is sufficient to reject the frozen candidate under its
prospective gate, not to estimate universal matchup strength.
