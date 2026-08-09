# Target-80 learned public-value plan V2 result

Completed: 2026-08-05

Status: **rejected; exact candidate closed; no package or submission**

## Identity and protocol

The evaluated candidate is exactly the Wobbuffet-free, three-Murkrow build
locked in `TARGET80_LEARNED_PUBLIC_VALUE_PLAN_V2_BUILD_LOCK.md`. Its model,
deck, resolver, and configuration hashes did not change after the lock. The
prospective gates are in `TARGET80_LEARNED_PUBLIC_VALUE_PLAN_V2_PROTOCOL.md`.

The learned model cleared every frozen offline gate. The official-engine smoke
then terminated normally in 110 decisions and emitted three successful
resolver searches with real official-engine branch execution, the declared
public-range hidden-state abstraction, the learned V2 leaf, and no error.

## Sealed development result

All four cells completed before any outcome was inspected.

| Frozen row | V2 | Same-deck control |
|---|---:|---:|
| AshV178MarnieFullBest, 12 games | 5-7-0 | 6-6-0 |
| SparkStarmieFroslassCFV13, 8 games | 3-4-1 | 2-5-1 |
| Total, 20 games each | **8-11-1** | **8-11-1** |

Draws are non-wins. All 40 games ended by the official terminal condition.

V2 emitted 58 completed searches across 12 games, changed 11 blueprint
actions, marked all 58 rows as official-engine rollout execution with the
`learned_public_value_v2` abstraction leaf, and recorded zero resolver errors.
Thus the runtime, search-count, activation, change-count, and error gates
passed. The strength gates failed:

- required at least 15/20 wins; observed 8/20;
- required at least four more wins than control; observed a tie; and
- required no row regression; V2 was one win worse against Marnie.

## Conclusion

Held-out outcome prediction did not become reliable counterfactual action
selection. The model was trained on states reached by V5, not on paired action
interventions; good trajectory AUC therefore did not prove that differences
between search leaves were calibrated action advantages. The official search
machinery was healthy and active, but its 11 confident changes produced no
aggregate win gain.

Do not tune its threshold, particle bound, activation routes, search depth, or
third-Murkrow deck after observing this panel. Do not package or submit this
identity. Any future learned resolver must learn paired counterfactual
advantages or counterfactual values from official-engine branches, with a new
prospective split and untouched evaluation. This result does not invalidate
the reusable official-engine adapter, telemetry, replay capture, or safe
blueprint fallback.

Evidence:

- `work/target80_public_value_plan_v2_smoke.jsonl`
- `work/target80_public_value_plan_v2_smoke.logs/`
- `work/target80_public_value_plan_v2_dev/`
- `work/target80_public_value_v2_training_report.json`

No Kaggle or internal-arena submission was made.
