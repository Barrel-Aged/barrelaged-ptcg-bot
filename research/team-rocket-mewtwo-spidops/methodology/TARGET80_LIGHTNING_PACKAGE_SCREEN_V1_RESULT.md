# Target-80 zero-Wobbuffet Lightning package screen V1 result

Status: **completed; both exact packages rejected**  
Completed: 2026-08-05 (America/Denver)

## Outcome

Neither queued Lightning package advances to exact-deck learning.

| Arm | Record | In play | Ready | Attack games | Attacks | 200+ damage | Median first attack |
|---|---:|---:|---:|---:|---:|---:|---:|
| Joltik / Galvantula route | **2-10** | 12/12 | 3/12 | 4/12 | 5 | 5 | 6.5 |
| Rocket Zapdos route | **2-10** | 11/12 | 8/12 | 8/12 | 16 | 10 | 5.0 |
| Rocket Zapdos deck control | **4-8** | 7/12 | 1/12 | 4/12 | 5 | 3 | 8.5 |

All 36 locked panel games ended normally in the official engine. Both decks
contained exactly 60 cards, zero Wobbuffet, and the frozen exact-Hancang model.
No engine, legality, import, timeout, or step-limit error occurred in the valid
Torch-runtime smokes or locked panel.

## Mechanism findings

The official card semantics were genuine and useful:

- `Jolting Charge` legally searched and attached Basic Grass and Lightning
  Energy.
- type 9 is the official evolution action used to put Galvantula ex on Joltik;
  the corrected V1R wrapper recognized it.
- `Charged Web` produced five 200-plus-damage events, but only in four games,
  all against MistFlat. It never attacked against Froslass and reached a ready
  state in only 3/12 games.
- One Basic Lightning plus one Team Rocket's Energy legally readied `Wicked
  Thunder`, and Team Rocket's Energy activated the attack's damage bonus.
- Maximum Belt plus both Energy types was present for two Wicked Thunder
  attacks. The corrected one-game smoke executed the full route on turn 3 and
  won, proving the intended breakpoint sequence can occur.

Mechanics did not convert into robust wins. The Galvantula line was too slow
and fragile, especially from the second seat and against Froslass. Zapdos was
far more reliable—attack games covered both seats (5 first, 3 second) and both
opponents (5 MistFlat, 3 Froslass)—but the routed policy still lost ten games
and underperformed the same-deck neural control by two wins. A one-shot
Lightning weakness conversion did not solve the full prize race or backup-
attacker problem.

## Gate decision

### Joltik / Galvantula

- integrity: **pass**;
- in-play and ready floor: **fail** (12/12 in play, 3/12 ready; required 9 and
  7);
- attack coverage: **fail** (4/12, only one second-seat game, zero Froslass
  games; required 6/12, two per seat, and two per opponent);
- impact: **pass** (five qualifying attacks);
- win floor: **fail** (2/12; required 5/12).

### Team Rocket's Zapdos

- integrity: **pass**;
- in-play and ready floor: **pass** (11/12 and 8/12);
- attack coverage: **pass** (8/12, both seats and both opponents);
- impact: **pass** (ten qualifying attacks and two Belt-plus-Rocket-Energy
  attacks);
- absolute/comparison gate: **fail** (2/12 versus the required 5/12 and versus
  control's 4/12).

The preregistered decision is therefore `close_both_packages`. Thresholds were
not changed after results. Do not train, route, package, submit, or rerun either
exact list unchanged.

## Implementation correction boundary

The first valid terminal smokes exposed wrapper behavior before the locked
panel. Galvantula was in hand but its official type-9 evolution action was not
recognized; Zapdos was charged but the neural fallback spent Switch while
Zapdos was already Active. V1R changed only those implementation mechanics and
created new immutable route directories with unchanged decks and model. The
locked panel used V1R only.

Two earlier one-step launches under the lab's nominal Python failed because
that interpreter lacked PyTorch. They are retained as environment provenance,
not counted as official games. All valid work used the installed Python 3.12
Torch runtime.

## Evidence and identities

- audit: `work/target80_lightning_screen_v1/analysis.json`;
- audit SHA-256:
  `5c5f9eb816fb95be7bbd6b16a50eac5ba2f73da8f847768bc38677a064528b8a`;
- Galvantula deck SHA-256:
  `b7b649ba120a17680ed62c5910539363ee66a0fe86ff68e779492bef632a63bc`;
- Galvantula V1R main SHA-256:
  `bf6e7d4094a2a72a8687c25da466df8a168331c88cb904bf362385c230d247a9`;
- Zapdos deck SHA-256:
  `b4d4e87fd490c404dc1457048f13c04fb1ba9b43fce0231a7092bf1b25f342bd`;
- Zapdos V1R main SHA-256:
  `faea4dae855e1e3614fba48529c6f2df07a7d2519821d5fc0f2d8c99f607e9ed`;
- frozen model SHA-256:
  `b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f`.

The official engine supplied legality, transitions, damage, knockouts, prizes,
and terminal outcomes. The deterministic route, thresholds, and package
selection are research abstractions. No package was built and no internal-arena
or Kaggle submission occurred.
