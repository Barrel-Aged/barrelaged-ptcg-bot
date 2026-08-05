# V1 Persistent Plan Validation Lock

Locked before any V1 outcome collection.

Protocol: `rocket-persistent-plan-causal/v1`  
Plan library: `artifacts/team_rocket_recovery/persistent_plan_v1_locked_library.json`  
Candidate: `locked_admitted_axes_blend_v1`  
Control: `zero_blueprint`

## Immutable vector

| Axis | Weight |
|---|---:|
| `charging_fuel` | 1.0 |
| `spidops_engine` | 1.0 |
| `tool_assignment` | 0.5 |
| `attack_tempo` | 0.25 |
| all other axes | 0.0 |

## Immutable execution

- Frozen Team Rocket deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`
- Frozen blueprint model SHA-256: `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`
- Persistent runtime SHA-256: `f6c3e2056eb143b6ad4bf537095aeb5b78703b0c9f4ffd789587279e8e1b85af`
- Locked plan-library SHA-256: `0a2ba32418abe558a13ecf60cee3798bb9d2ff0ef50d03f6e1635de41b44b043`
- D2 analysis SHA-256: `cda3ebbce53f7c075dfb85bffac2bfa773337bfd691071684308d26712017ba8`
- Opponents: `AshMarnieGeneralV110_extracted` and `HelbrosMarnieGeneralRecoveryV4_extracted`
- Team Rocket is seat 1 and every accepted game must declare seat 0 first.
- Complete official-engine games; ordinary redacted observations only; no visualizer truth or search branches.

## Sample and seeds

Target: 4,000 accepted validation games.

- 1,000 games per arm against V110.
- 1,000 games per arm against Recovery.
- Eight independent 125-games-per-arm shards per opponent.
- V110 seeds: `774101` through `774108`.
- Recovery seeds: `774201` through `774208`.
- Analysis bootstrap: 50,000 stratified replicates, seed `774999`.

No optional stopping, mid-run inspection, arm changes, seed changes, or sample-size changes are allowed.

## Locked gate

Validation passes only if:

1. equal-opponent point improvement is at least +3 percentage points;
2. stratified 95% bootstrap lower bound is greater than 0;
3. neither opponent's point estimate is worse than control by more than 2 points;
4. at least 10% of candidate games contain an action intervention;
5. there are zero illegal actions, zero overlay exception fallbacks, and zero hidden-information violations.

If any condition fails, no persistent-plan test is opened, no candidate is packaged, and no arena submission is made from this family.
