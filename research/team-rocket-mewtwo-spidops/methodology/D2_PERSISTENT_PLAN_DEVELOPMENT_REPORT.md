# D2 Persistent Plan Development Report

Protocol: `rocket-persistent-plan-causal/v1`  
Split: D2 combination development  
Games: 3,840 complete local official-engine games  
Allocation: 160 games per arm against each of two top Marnie opponents  
Correctness: zero illegal actions and zero overlay exception fallbacks

## Result

The D1 `charging_fuel_100` point estimate did not reproduce by itself: it moved from +8.125 percentage points in D1 to -0.3125 points in D2. This prevents the discovery winner from being promoted on a lucky first sample.

The D2 winner was the preregistered `admitted_axes_blend`, which combines only the four axes admitted by D1:

- `charging_fuel = 1.0`
- `spidops_engine = 1.0`
- `tool_assignment = 0.5`
- `attack_tempo = 0.25`
- every other plan weight is zero

The zero blueprint won 39.375% on an equal-opponent basis. The blend won 43.75%, an improvement of 4.375 percentage points. It improved V110 by 1.25 points and Recovery by 7.5 points. Its stratified 95% bootstrap interval is `[-3.125, +12.1875]` percentage points and therefore still includes zero. It intervened in 63.4% of games and had zero correctness failures.

The next most positive arms were `tool_assignment_100` (+4.0625), `charging100_tool100` (+3.75), `spidops_engine_100` (+3.4375), `charging_fuel_050` (+3.4375), and `attack_tempo_100` (+3.125). Broadly increasing strength did not monotonically help: `charging100_spidops100` was -2.5 points, and `charging_fuel_150` was only +0.3125.

## Decision

Exactly one vector is locked for V1: `locked_admitted_axes_blend_v1`, identical to the D2 winner above. This is a development selection, not a validation or arena strength claim. The D2 analysis file is `artifacts/team_rocket_recovery/persistent_plan_d2_v1/analysis.json`, SHA-256 `cda3ebbce53f7c075dfb85bffac2bfa773337bfd691071684308d26712017ba8`.

