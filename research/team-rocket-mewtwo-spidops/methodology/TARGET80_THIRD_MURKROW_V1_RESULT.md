# Target-80 third-Murkrow V1 result

Recorded: 2026-08-05

Status: **deck-only gate failed; exact substitution not advanced**

## Frozen result

All 24 official-engine games were terminal, with zero draws, crashes, illegal
selections, or timeouts.

| Arm | W-L vs `AshV178MarnieFullBest` |
|---|---:|
| `ThirdMurkrowV1` | 6-6 |
| contemporaneous immutable V5 | 3-9 |

The candidate beat the control by three wins but failed the required absolute
9-3 floor. The protocol requires every gate, so no broad panel, packaging, or
submission is authorized.

## Mechanism audit

- Murkrow reached play in 11/12 games, passing the six-game availability gate.
- It used `Deceit` on 26 distinct game-turns across five games.
- It never used `Torment`.
- The five Deceit games went 2-3; all six candidate wins included Murkrow in
  play, but that descriptive association is not a causal estimate.

The extra copy is operationally compatible with the frozen policy and was far
less damaging than the tested Kangaskhan, Galvantula, Grimer, or lock packages.
It nevertheless did not establish target-level strength. Repeated zero-damage
Deceit turns also show that increased Supporter access alone does not solve the
prize-race and ready-attacker deficit.

## Decision

Do not broaden, package, or submit the exact deck-only candidate. Do not add a
post-hoc forced-Torment rule from these outcomes. The zero-Wobbuffet,
three-Murkrow list may be retained only as a research substrate for a
materially new, prospectively frozen strategic-planning hypothesis; any such
hypothesis needs fresh evidence and cannot pool this panel.

The live V5 package remains unchanged.

## Durable evidence

- protocol: `TARGET80_THIRD_MURKROW_V1_PROTOCOL.md`;
- candidate: `work/target80_candidates/murkrow_three_v1/`;
- smoke: `work/target80_third_murkrow_smoke_v1.jsonl`;
- frozen panel and traces: `work/target80_third_murkrow_explore_v1/`.

Evidence boundary: all legality, public card-use, and outcome statements come
from official-engine execution. This small screen is a development result, not
a universal matchup estimate or a strength claim.
