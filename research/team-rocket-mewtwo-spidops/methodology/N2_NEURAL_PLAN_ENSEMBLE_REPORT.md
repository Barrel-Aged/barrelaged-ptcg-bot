# N2 Neural Plan Ensemble Report

Status: development gates failed; no validation, test, packaging, or submission  
Protocol: `rocket-neural-plan-ensemble/v1`  
Analysis SHA-256: `24e76285ee8e770170505875e0ff2e66262f2859422491a98d187d8dea40d0fb`

N2 collected 2,880 fresh complete official-engine games, exactly 240 per arm per opponent. Every game placed Team Rocket second; exception fallbacks were zero.

| Arm | Equal-opponent difference | 95% bootstrap interval | V110 difference | RecoveryV4 difference | Intervention games |
|---|---:|---:|---:|---:|---:|
| `mean_logits_025` | +4.38 pp | [-1.88, +10.63] | -3.33 pp | +12.08 pp | 49.38% |
| `mean_logits_050` | +3.54 pp | [-2.71, +9.79] | -2.50 pp | +9.58 pp | 73.96% |
| `consensus_4of4` | +3.13 pp | [-2.92, +9.38] | -2.92 pp | +9.17 pp | 49.38% |
| `consensus_3of4` | +1.04 pp | [-5.21, +7.08] | -3.33 pp | +5.42 pp | 76.67% |
| `mean_logits_100` | +0.21 pp | [-5.83, +6.46] | -2.08 pp | +2.50 pp | 89.17% |

No arm passed the locked requirement for a strictly positive lower bound and no worse than -2 points against either opponent. N2 therefore stops before validation.

The consistent interaction is useful: conservative residuals improved strongly against `HelbrosMarnieGeneralRecoveryV4` and regressed against `AshMarnieGeneralV110`, despite the two opponents sharing the same deck. N3 may test a separately trained public-history opponent-style classifier and a fail-closed exploitation overlay: use the conservative Recovery policy only at high classifier confidence, otherwise use the exact blueprint. That overlay must be trained and evaluated using public/own-visible observations only and must pass a full mixed-opponent non-regression validation before deployment.
