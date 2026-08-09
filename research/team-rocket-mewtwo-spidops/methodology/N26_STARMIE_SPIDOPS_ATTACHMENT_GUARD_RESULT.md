# N26 Starmie Spidops attachment guard result

Status: **development failure; rejected**  
Completed: 2026-08-01

## Hypothesis

After Staryu or Mega Starmie ex becomes public, redirect selected manual Energy
attachments from Mewtwo to an unready Spidops when a narrow readiness rule
prefers Spidops. The proposed mechanism was Grass-weakness prize conversion:
Rocket Rush can one-shot Mega Starmie ex after sufficient Team Rocket board
development.

The guard used only public active, Bench, and discard cards. Unit tests verify
that opponent hand, prize, and deck contents cannot trigger it. It returned
only an offered official-engine option and preserved the unchanged neural
policy outside its narrow trigger.

## Development result

Exact official-engine opponent bundle:
`SparkStarmieFroslassCFV13`.

| Arm | Games | W-L-D | Raw win rate |
|---|---:|---:|---:|
| untouched ScaleStopV3 control | 40 | 17-21-2 | 42.5% |
| balanced attachment guard | 40 | 9-31-0 | 22.5% |

The two panels were independent rather than paired, so the exact difference is
not a causal estimate. The candidate result is nevertheless far too weak to
justify broader testing or deployment, especially in light of the 699-game
historical Starmie evidence. The apparent attachment correlation did not
transfer into a safe intervention.

## Decision

The attachment guard is rejected. It is not packaged, submitted, or installed
as a live bot. The immutable ScaleBCV2 and ScaleStopV3 arena identities remain
unchanged. Any future Starmie work must use a new disclosed hypothesis and an
independent evidence boundary; this exact post-filter is closed.
