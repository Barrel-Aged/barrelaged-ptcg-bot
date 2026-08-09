# N11 Fresh Epoch-4 Confirmation Result

Status: **rejected at the locked confirmation gate; live bot unchanged**  
Protocol: `rocket-n11-fresh-epoch4-confirmation/v1`  
Completed: 2026-08-01

## Execution

N11 replaced the operationally invalid N10 attempt with an entirely fresh,
uninterrupted panel. It fixed the single N9 epoch-4 candidate before
collection, used new seeds 800101, 800102, 800201, and 800202, and excluded
every N10 row.

All four strict official-engine collectors exited zero. Each shard contained
exactly 400 control and 400 candidate games, producing 3,200 total rows and
800 games per arm/opponent cell. The analyzer verified exact model, deck, arm,
collector, opponent, seat, and row-count provenance and found no correctness
or fallback failure.

## Locked result

| Opponent | Control | Candidate | Difference | Candidate 95% bootstrap interval |
|---|---:|---:|---:|---:|
| `AshMarnieGeneralV110` | 41.250% (330/800) | 41.625% (333/800) | +0.375 pp | [-4.50, +5.13] pp |
| `HelbrosMarnieGeneralRecoveryV4` | 40.750% (326/800) | 40.625% (325/800) | -0.125 pp | [-5.00, +4.63] pp |
| Equal-opponent aggregate | — | — | **+0.125 pp** | **[-3.25, +3.50] pp** |

The preregistered gate required at least +2.0 points aggregate, a strictly
positive point estimate against each opponent, and an aggregate 95% lower
bound above zero. The candidate failed all three strength conditions while
passing correctness.

Immutable analysis:
`artifacts/team_rocket_recovery/n11_fresh_epoch4_confirmation/panel/analysis.json`,
SHA-256
`7c6e0a5f2a9c78e35902f3183be8e83c1d439a26aa251a216c0e46a0e22b7b27`.

## Interpretation

The much larger independent sample resolves N9's encouraging epoch-4
development estimate as a false positive. The 385-parameter balanced AWR
action-head update is effectively neutral against these two opponents and is
not a credible route to rank one.

No N9/N10/N11 checkpoint may be validated further, packaged, submitted, or
used for a strength claim. The next experiment must change the learning signal,
not tune another scalar, checkpoint, or sample around this rejected update.
It should learn decision-level counterfactual credit from official-engine
branches while retaining a frozen baseline safety constraint, then face wholly
fresh development and confirmation panels.

