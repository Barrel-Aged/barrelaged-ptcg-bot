# Target-80 Budew / Arbok bridge exploratory result

Recorded: 2026-08-04

Status: **branch rejected; no package or arena submission authorized**

## Hypothesis

The branch tested a Wobbuffet-free bridge against the current Marnie policy:
Budew would delay Item-driven setup with `Itchy Pollen`, Team Rocket's Arbok
would then prevent non-Rocket Pokemon with Abilities from entering play, and a
prepared Spidops would convert that temporary lock into prizes.  All tests used
the official game engine and public/own-visible information only.

## Frozen exploratory results

| Candidate | Material change | Ash games | W-L |
|---|---|---:|---:|
| `budew_earlypad_control_v1` | Wobbuffet -> one Budew | 12 | 4-8 |
| `arbok_budew_bridge_v3` | one Budew added to the earlier Arbok shell | 12 | 6-6 |
| `arbok_budew_energy_v4` | explicit Darkness-unit charging for Arbok | 12 | 4-8 |
| V4 deck-identical old-policy control | same 60 cards, no charging override | 12 | 8-4 |
| `arbok_budew_release_v6` | two Budew, two Switch, certified lethal release | 12 | 3-9 |
| V6 deck-identical old-policy control | same 60 cards, no release override | 12 | 5-7 |
| V5 live-policy control in the V6 screen | unchanged V5 | 12 | 8-4 |

All 84 listed games terminated normally with zero draws.  The small panels are
development screens rather than estimates of live-arena strength, but the V6
candidate missed its prospective advancement threshold by a wide margin and
lost to both controls.

## Mechanism checks

- Direct Budew was visible in eight games, became Active in three, and used
  `Itchy Pollen` in all three; those three games were losses.
- The original Arbok policy accidentally treated expanded energy units as
  physical cards.  V4 corrected that bookkeeping and successfully charged and
  attacked with Arbok, but diverting attachments to `Spinning Tail` weakened
  the primary Spidops prize plan.
- A two-Budew build established Arbok on turn 3 and delayed Grimmsnarl until
  turn 24 in its smoke, but produced a permanent-lock stalemate rather than a
  prize lead.
- V6 added a public-state certificate: leave Arbok only if a benched Spidops
  already had legal energy and enough computed `Rocket Rush` damage to knock
  out the opposing Active.  Its corrected smoke genuinely used `Spinning
  Tail`, retreated to Spidops for a knockout attempt, later used Switch, and
  attacked again.  That smoke still ended behind 3 prizes to 1, and the frozen
  screen then scored 3-9.

## Decision

Close the exact Budew one-slot, Arbok/Budew bridge, Arbok charging, two-Budew
permanent-lock, and certified-release variants.  Do not scale them, package
them, or submit them.  The evidence supports the narrower strategic lesson
that Arbok can create genuine tempo but is an expensive temporary wall, not a
reliable primary attacker or an 80%-target solution for this Spidops shell.

The live `HelbrosRocketSpidopsEarlyPadLethalV5` identity and its archived
package remain byte-unchanged.

## Durable evidence

- candidates: `work/target80_candidates/budew_earlypad_control_v1/`,
  `work/target80_candidates/arbok_budew_bridge_v3/`,
  `work/target80_candidates/arbok_budew_energy_v4/`,
  `work/target80_candidates/arbok_budew_two_v5/`, and
  `work/target80_candidates/arbok_budew_release_v6/`;
- V6 corrected smoke:
  `work/target80_arbok_budew_release_smoke_v6b.jsonl`;
- frozen V6 three-arm screen:
  `work/target80_arbok_budew_release_explore_v6/`.

Evidence boundary: official-engine execution proves legality and observed
behavior.  It does not prove hidden-state beliefs, causal superiority outside
the frozen comparisons, or live-arena performance.
