# V6 public-belief safe-plan resolver result

Status: **development rejected; no confirmation, package, or submission**  
Completed: 2026-08-04

## Outcome

The depth-limited safe resolver passed runtime and behavioral gates but failed
the frozen strength gates. It therefore remains a development artifact and
cannot replace immutable V5.

| Opponent | V5 control | SafePlan V1 | Difference |
|---|---:|---:|---:|
| AshV178MarnieFullBest | 3-9 | 5-7 | +2 wins |
| SparkStarmieFroslassCFV13 | 3-5 | 1-7 | -2 wins |
| SparkStarmieMistFlatV20 | 1-7 | 3-5 | +2 wins |
| SoldierPureRLV45 | 5-3 | 5-3 | 0 wins |
| **Total** | **12-24** | **14-22** | **+2 wins** |

All 72 games ended normally. The frozen gate required at least three aggregate
wins over control and no hard-matchup row more than one win worse. V1 failed
both: the aggregate gain was two wins and CF13 regressed by two.

## Resolver behavior

Across the 36 V1 games, the official public-belief resolver completed 95
invocations, changed 26 baseline actions, and emitted zero runtime, search, or
legality errors.

| Cell | Searches | Changes |
|---|---:|---:|
| Marnie | 36 | 11 |
| Starmie CF13 | 24 | 7 |
| Starmie Mist20 | 24 | 7 |
| Soldier | 11 | 1 |

The smoke-game changes were strategically interpretable—evolving Spidops
before an energy commitment and attaching to Mewtwo before a redraw/stadium
sequence—but the fixed leaf model did not generalize consistently across the
two Starmie policies. Baseline safety over sampled particles was not sufficient
to guarantee match-level safety under policy/model mismatch.

## Decision boundary

No parameter was tuned from these results. No untouched confirmation is
opened, and no archive is built or submitted. The exact V1 family is closed;
rescaling its leaf weights or margins post hoc is not authorized by this
panel. A credible successor needs either learned matchup-conditioned public
values with held-out calibration or a materially different deck/strategy
engine, not another manual reweighting of this sampled resolver.

Artifacts:

- frozen protocol: `work/target80_candidates/SAFE_PLAN_RESOLVER_PROTOCOL.md`
- candidate: `work/target80_candidates/earlypad_bangle_safe_plan_v1/`
- smoke: `work/target80_safe_plan_smoke/`
- frozen panel: `work/target80_safe_plan_dev_v1/`

