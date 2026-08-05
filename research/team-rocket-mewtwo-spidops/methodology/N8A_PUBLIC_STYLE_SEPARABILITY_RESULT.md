# N8A Public-Style Separability Result

Status: **rejected as an unsafe/ineffective router; sealed validation unopened**  
Completed: 2026-07-30

## Purpose

N7 found a V110-specific policy exploit that regressed badly against Recovery
V4. N8A tested whether a stronger public-history classifier could identify
V110-like behavior safely enough to expose that exploit while keeping the live
policy as a Recovery-safe baseline.

This was a separability diagnosis, not a policy-strength experiment. It reused
only the 400 training and 200 calibration trajectories from N4. The 400
previously sealed N4 validation trajectories remain unopened.

## Information boundary and method

Every input was derived from public opponent events, public board cards, hand
and deck counts, prizes, and turn number. No opponent hand, deck order, prize
identity, engine-private state, implementation name, or future event entered a
classifier.

The old N4 linear model may have been underpowered, so N8A compared:

- a 1,933-parameter normalized linear token classifier;
- an 88,009-parameter nonlinear token MLP; and
- a 94,377-parameter projected GRU over the full public sequence.

Within the old training split, 300 games fitted model parameters and 100 games
selected a consecutive-evidence threshold. The threshold was required to keep
Recovery V4 false activation at or below 1%. The untouched 200-game N4
calibration split then served as a diagnostic holdout. Positive activation
means “use the V110 exploit”; a Recovery activation is therefore the safety
error.

## Result

| Model | Final-step AUC | Recovery activations | V110 activations | Median V110 activation decision |
|---|---:|---:|---:|---:|
| Linear | 0.594 | 4/100 | 7/100 | 35.0 |
| MLP | 0.599 | 0/100 | 0/100 | — |
| GRU | 0.581 | 0/100 | 2/100 | 34.5 |

The linear model violated the intended Recovery safety boundary. The two safe
nonlinear models activated in zero or two percent of V110 games, and the rare
GRU activations occurred around decision 35—too late and too infrequent to
recover N7's matchup gain. More capacity reduced training loss but did not
create actionable out-of-sample information.

## Decision

The public-observation process does not identify these two near-sibling greedy
policies reliably enough for a safe implementation-identity router. N8A is
closed. No N4 validation data was opened, no gated actor was built, no strength
panel was run, and nothing was packaged or submitted.

The robust alternative is to learn or transfer a policy change that helps both
opponents. The next preregistered experiment, N8B, tests the minimal action-head
delta that separates the rank-leading Recovery V4 policy from its otherwise
parameter-identical V110 sibling, transplanted onto the Rocket-adapted
blueprint.

Artifact:
`artifacts/team_rocket_recovery/n8_safe_style_overlay/public_style_separability.json`
(SHA-256
`50ff743ce084a9093ac8fa9c89a1cc375d1f6412f5fab7adc1332c3f376703e4`).

