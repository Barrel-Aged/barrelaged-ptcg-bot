# N18 Cross-Deck Leader Transfer Result

Status: **development rejected; no confirmation; no deployment**  
Protocol: `rocket-n18-cross-deck-leader-transfer/v1`

## Decision

No N18 challenger passed the preregistered development gates.  No candidate was
selected, the 3,200-game confirmation is not authorized, and no bundle may be
submitted or substituted for the live bot.  The unchanged live submission
remains `HelbrosRocketSpidopsScaleBCV2`, model SHA-256
`3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.

## Correctness

All six locked shards sealed before analysis:

- 2,880 genuine native official-engine games;
- 720 games per arm;
- 120 games in every arm-by-opponent-by-Rocket-seat cell;
- three frozen opponents and both Rocket seats;
- exact Team Rocket deck SHA-256
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- isolated packaged-agent execution for every arm and opponent;
- zero illegal actions, exception fallbacks, count errors, hash errors, engine
  errors, or provenance errors.

The analyzer was the first process permitted to read outcomes after all six
JSONL files and all six manifests sealed.

## Aggregate result

| Arm | Wins | Draws | Games | Win rate | Improvement vs control |
|---|---:|---:|---:|---:|---:|
| `control` | 317 | 0 | 720 | 44.03% | — |
| `live_residual` | 306 | 1 | 720 | 42.50% | −1.53 pp |
| `leader_base` | 198 | 0 | 720 | 27.50% | −16.53 pp |
| `leader_residual` | 236 | 1 | 720 | 32.78% | −11.25 pp |

The ordinary and multiplicity-adjusted intervals were:

| Challenger | Ordinary 95% interval | Bonferroni-simultaneous 95% interval |
|---|---:|---:|
| `live_residual` | [−6.53, +3.61] pp | [−7.64, +4.72] pp |
| `leader_base` | [−21.39, −11.67] pp | [−22.36, −10.56] pp |
| `leader_residual` | [−16.25, −6.25] pp | [−17.27, −5.14] pp |

## Cell results

Wins out of 120 games in each cell:

| Opponent / Rocket seat | Control | Live + residual | Leader base | Leader + residual |
|---|---:|---:|---:|---:|
| current leader / seat 0 | 59 | 58 | 33 | 41 |
| current leader / seat 1 | 52 | 46 | 23 | 35 |
| RecoveryV4 / seat 0 | 54 | 51 | 35 | 44 |
| RecoveryV4 / seat 1 | 40 | 40 | 32 | 32 |
| V110 / seat 0 | 61 | 65 | 41 | 42 |
| V110 / seat 1 | 51 | 46 | 34 | 42 |

`leader_base` was negative against every opponent aggregate and both seats:
−22.92 pp against the current leader, −11.25 pp against RecoveryV4, and
−15.42 pp against V110.  `leader_residual` was also negative everywhere:
−14.58, −7.50, and −11.67 pp respectively.

`live_residual` was closer but still failed every directional aggregate gate:
−2.92 pp against the current leader, −1.25 pp against RecoveryV4, and −0.42 pp
against V110; seat 0 was exactly flat and seat 1 was −3.06 pp.  Its worst cell
was exactly −5.0 pp, and its simultaneous lower bound was far below the locked
−0.5 pp threshold.

## What N18 establishes

The current leader's strong official-arena performance against the live Rocket
bot does not transfer by simply replacing the live model's compatible
last-block/action-head weights.  The direct transfer produced a large, precise
regression on the exact Team Rocket deck.  The leader's recurrent residual
recovered about 5.28 pp relative to that bad base but remained 11.25 pp below
the live control.  Adding only the residual to the live model was not useful.

Accordingly, the following N18 family is closed and must not be resampled,
scaled, seed-tuned, opponent-tuned, checkpoint-selected, or promoted:

- raw leader-model transfer to Team Rocket;
- raw leader history-residual transfer to the live model;
- raw leader model plus leader residual transfer.

This does not show that fresh leader trajectories contain no useful training
signal.  It shows that the leader's deck-specialized parameters and recurrent
overlay are not a safe plug-in policy for Team Rocket.  A future experiment
must preserve the live Rocket policy and learn only through a separately
preregistered, retention-constrained objective or acquire new exact-Rocket
counterfactual evidence.  It may not reinterpret these negative arms as
positive teachers.

## Durable artifacts

- protocol SHA-256:
  `b64bd56fb0c1593f2310b636723c83c7c08596f5dbb63c3d9e848015b3c25805`;
- execution lock SHA-256:
  `c1635ac4798432fc6ada2e4ededc8da62d148d8b6c2a358cddfc8a44bf5309fe`;
- collector SHA-256:
  `2891a7c6ec14e7c6de68f755b66e5ede606bcf302ac23e01254b4bcc02844625`;
- analyzer SHA-256:
  `5b9f8b0f9379ebf31306b2b8099b48456a5486313231e4b19dd9e83bcca68ca1`;
- machine-readable analysis:
  `artifacts/team_rocket_recovery/n18_new_leader_evidence/development_panel/analysis.json`;
- analysis SHA-256:
  `d8548cd8c90f4591ddb5de9585a274ff7ce3faa56b6c7ae4a1f61adafd140296`.

## Claim boundary

These are official-engine development results for the frozen exact-deck arms
and opponents only.  They authorize rejection, not a claim about every form of
leader-guided learning.  No confirmation, leaderboard submission, rank
improvement, superhuman performance, or deployment claim is made.
