# Target-80 Golbat / Spidops Bite Single V2 — Locked Result

Decision: **rejected; do not package or submit this identity.**

Protocol: `TARGET80_GOLBAT_SPIDOPS_BITE_SINGLE_V2_PROTOCOL.md`

Panel directory: `work/target80_golbat_single_v2_locked_panel_20260804`

## Exact official-engine result

All 120 preregistered games completed after all eight arms were launched from
the frozen checkpoint. Every record was terminal, all eight stderr files were
empty, all traces were present, and there were no draws.

| Opponent | Golbat V2 | V5 control | Difference |
|---|---:|---:|---:|
| Ash Marnie | 6-18 | 14-10 | -8 wins |
| Starmie CF13 | 1-11 | 2-10 | -1 win |
| Violet Alakazam | 7-5 | 12-0 | -5 wins |
| V5 Rocket mirror | 6-6 | 4-8 | +2 wins |
| **Total** | **20-40** | **32-28** | **-12 wins** |

Candidate seat split:

- Marnie: first seat 4/12, second seat 2/12.
- Starmie: first seat 0/6, second seat 1/6.
- Alakazam: first seat 4/6, second seat 3/6.
- Mirror: first seat 2/6, second seat 4/6.

## Admission gates

1. Normal completion: **pass** — 120/120 terminal, zero stderr bytes, zero
   draws.
2. Overall strength: **fail** — 20/60, below the required 34/60 and 12 wins
   behind control instead of at least four ahead.
3. Marnie conversion: **fail** — 6/24, below 13/24 and eight wins behind
   control.
4. Non-Marnie robustness: **fail** — Starmie was 1/12, and the combined
   non-Marnie score was 14/36 versus control 18/36 (four behind; limit two).
5. Mechanism/routing: **fail overall**. Sneaky Bite did activate in 16/24
   Marnie games, exceeding the minimum eight, but the public archetype gate
   included Froslass (104). That card is also present in Starmie, so the
   supposedly Marnie-only overlay can route on a non-Marnie deck. Golbat
   activation was observed in 2/12 Starmie games, 6/12 Alakazam games, and
   5/12 mirror games; neural play can account for some non-Marnie activations,
   but the Starmie public-ID collision proves the override itself was not
   exclusive.
6. Integrity: **pass** — hashes reproduced exactly; 60 cards, zero Wobbuffet,
   one Zubat, and one Golbat.

## Interpretation

The exploratory V1 result (6-6 while its small control happened to go 4-8) was
noise. The larger locked panel reverses it decisively. The Golbat mechanism is
engine-real and frequently fires, but its deck slots, bench slot, forced setup,
and out-of-distribution neural decisions cost much more than the 20-damage
threshold gains. Even in the intended Marnie matchup it lost heavily to the
unchanged control.

The public detector's Froslass collision is a separate implementation lesson:
future archetype routing must require an archetype-unique core signature or a
multi-card public condition. Fixing that detector would not rescue this
candidate's 6-18 intended-matchup result.

## Closure

Close the Golbat/Sneaky Bite family for this deck effort. Preserve the files as
negative evidence. Do not alter or replace the live V5 arena bot from this
result, and do not pool these games into any future identity's 200-game target.

