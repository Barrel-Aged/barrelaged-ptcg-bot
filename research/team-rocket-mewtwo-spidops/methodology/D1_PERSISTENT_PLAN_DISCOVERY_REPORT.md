# D1 Persistent Plan Discovery Report

Protocol: `rocket-persistent-plan-causal/v1`  
Split: D1 discovery (development-only)  
Games: 2,560 complete local official-engine games  
Seat condition: Team Rocket in seat 1, seat 0 officially declared first  
Correctness: 2,560/2,560 terminal games; zero illegal actions; zero overlay exception fallbacks  
Frozen deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`  
Frozen blueprint SHA-256: `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`

## Result

The first whole-game plan experiment found a positive, coherent signal that the earlier single-action experiments did not find. The zero-plan blueprint won 47.5% against AshMarnieGeneralV110 and 37.5% against HelbrosMarnieGeneralRecoveryV4 (42.5% equal-opponent average). The best arm, `charging_fuel_100`, won 51.25% and 50.0% respectively (50.625% average): a stratified improvement of 8.125 percentage points.

This is discovery evidence, not a validation or arena strength claim. The stratified 95% bootstrap interval for the best arm is `[-2.50, +19.375]` percentage points, so it still includes zero.

| Plan | Average difference | V110 difference | Recovery difference | 95% bootstrap interval | Games with intervention |
|---|---:|---:|---:|---:|---:|
| `charging_fuel_100` | +8.125 pp | +3.75 pp | +12.50 pp | [-2.50, +19.375] pp | 18.1% |
| `spidops_engine_100` | +6.25 pp | +3.75 pp | +8.75 pp | [-5.00, +16.875] pp | 48.1% |
| `tool_assignment_100` | +5.00 pp | +2.50 pp | +7.50 pp | [-5.625, +16.25] pp | 9.4% |
| `attack_tempo_100` | +1.25 pp | -1.25 pp | +3.75 pp | [-9.375, +11.875] pp | 67.5% |
| `zero_blueprint` | 0 pp | 0 pp | 0 pp | control | 0% |

The broad coordinated plans were not automatically better. `coordinated_two_attacker` was -2.5 points, `coordinated_mewtwo_line` was -3.75, and `patient_full_setup` was -6.875. Direct `mewtwo_charge_100` was the worst arm at -10 points despite increasing the frequency of Mewtwo attacks. This is useful causal evidence: forcing visible Mewtwo progress is not itself the winning objective. The beneficial signal is narrower—use Spidops's free Energy recovery and preserve the engine's attack capacity while leaving tactical choices to the blueprint.

## D1 decision

Per the preregistered gate, at most four positive, non-fragile plans may advance. The four highest-ranked eligible axes are:

1. `charging_fuel`
2. `spidops_engine`
3. `tool_assignment`
4. `attack_tempo`

`mewtwo_access_100` is excluded because it regressed V110 by 7.5 points. `coordinated_spidops_line` met the broad per-opponent point constraint but ranked below `attack_tempo_100`, so it is excluded by the four-plan cap. All other axes and all broad D1 combinations are excluded from D2.

The immutable analysis is `artifacts/team_rocket_recovery/persistent_plan_d1_v1/analysis.json` with SHA-256 `76b63ea4ba6060cc13070811b6e7022718e841d95acd2d74e39a797f7eb17de8`.

