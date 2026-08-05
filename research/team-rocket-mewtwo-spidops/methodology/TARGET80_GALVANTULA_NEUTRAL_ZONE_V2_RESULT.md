# Target-80 Galvantula / Neutralization Zone V2 result

Recorded: 2026-08-04

Status: **rejected; complete Galvantula family closed**

## Frozen result

All 36 official-engine games were terminal, with zero draws, crashes, illegal
selections, or timeouts.

| Arm | W-L vs `AshV178MarnieFullBest` |
|---|---:|
| `GalvantulaNeutralZoneV2` | 2-10 |
| deck-identical `GalvantulaZoneControlV2` | 3-9 |
| immutable V5 | 9-3 |

V2 required at least 9-3 and a two-win lead over each control.  It missed all
outcome gates and finished one win below its deck-identical control.

## Mechanism audit

Neutralization Zone was an engine-real protection mechanism but not a reliable
plan:

- In game 10, the candidate played Zone immediately before `Shocking Web`.
  Two subsequent `Shadow Bullet` attacks recorded zero damage, Galvantula
  remained Active, made three 360-damage attacks, and won.
- In other games, Marnie commonly replaced Zone with Spikemuth Gym before the
  protected attack window, or the Galvantula line and stadium did not align.
- The prospective requirement for three distinct protected attack windows was
  not met.
- The one successful demonstration confirms the causal card interaction; the
  2-10 aggregate shows that a searchable singleton stadium is far too sparse
  and too easy for the four-Stadium opponent to counter.

## Decision

Close both Galvantula V1 and V2, including further line scaling, Lightning
tuning, and Neutralization Zone variants.  The family can produce spectacular
individual turns but cannot establish the repeatable board and prize continuity
required by the target.  Do not run a broad panel, package, submit, or pool any
of these games with another identity.

The live V5 package remains unchanged.

## Durable evidence

- protocol: `TARGET80_GALVANTULA_NEUTRAL_ZONE_V2_PROTOCOL.md`;
- candidate: `work/target80_candidates/galvantula_neutral_zone_v2/`;
- deck-identical control:
  `work/target80_candidates/galvantula_zone_control_v2/`;
- smoke: `work/target80_galvantula_neutral_zone_smoke_v2.jsonl`;
- frozen panel and traces:
  `work/target80_galvantula_neutral_zone_explore_v2/`.

Evidence boundary: all stated mechanics and outcomes are from official-engine
execution.  The development sample rejects the frozen candidate under its
prospective gate; it is not a universal matchup estimate.
