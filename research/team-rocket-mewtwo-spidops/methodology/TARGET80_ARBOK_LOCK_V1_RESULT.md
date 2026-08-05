# Target-80 Arbok Lock V1 result

Status: **development rejected; no broad panel, package, or submission**  
Completed: 2026-08-04

All three frozen 20-game official-engine arms completed with terminal results,
zero stderr, and no invalid action, exception, timeout, or step-limit result.

| Arm | W-L | Win rate | Seat 0 | Seat 1 | Mean steps |
|---|---:|---:|---:|---:|---:|
| untouched V5 control | 6-14 | 30% | 4-6 | 2-8 | 144.80 |
| Wobbuffet-free deck only | 8-12 | 40% | 4-6 | 4-6 | 155.60 |
| Wobbuffet-free public Arbok lock | 11-9 | 55% | 7-3 | 4-6 | 153.15 |

The public-lock arm improved by five wins over the untouched control and three
over the deck-only arm, so the mechanism was behaviorally useful. It failed the
frozen admission requirements of at least 16/20 overall and at least 7/10 from
each seat. Both exact challengers are therefore closed and must not enter the
broad panel, confirmation, packaging, or arena submission.

## Mechanism audit

Arbok became Active before any public Grimmsnarl ex in 14 games; those games
went 10-4 (71.4%) and averaged 3.5 Spinning Tail attacks. Five games established
Arbok only after Grimmsnarl was already public and went 1-4. One game never
established Arbok and lost. The seat split explains the remaining ceiling:
Arbok generally became Active on turn three when Rocket went first, but on
turn four when Rocket went second, often after Marnie had its first evolution
window.

This supports a new mechanism, not a retune of the rejected package: retain the
public Arbok evidence as an early first-seat plan, while adding a Basic wall
that remains effective after Grimmsnarl has evolved. Cornerstone Mask Ogerpon
ex is the relevant legal hypothesis because its Cornerstone Stance prevents
attack damage from Grimmsnarl ex, Froslass, and Munkidori, all of which have
Abilities. It requires its own frozen deck and policy experiment; it is not an
authorized post-hoc modification of Arbok V1.

## Artifacts

- protocol: `TARGET80_ARBOK_LOCK_V1_PROTOCOL.md`;
- full summaries and traces: `work/target80_arbok_development_v1/`;
- rejected deck-only candidate:
  `work/target80_candidates/arbok_lock_base_v1/`;
- rejected public-lock candidate:
  `work/target80_candidates/arbok_lock_guard_v1/`.

No Kaggle or internal-arena submission occurred.
