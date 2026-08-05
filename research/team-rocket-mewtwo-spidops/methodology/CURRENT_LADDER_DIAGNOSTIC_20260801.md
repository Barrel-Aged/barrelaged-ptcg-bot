# Current ladder diagnostic — 2026-08-01

Status: **diagnostic complete; frozen live identities unchanged**

## Why this slice was opened

`HelbrosRocketSpidopsScaleBCV2` fell from rank 1 at 92 games to rank 3–4
around 180 games. A bounded read-only replay pull and exact-current-leader
panel were used to distinguish a real policy regression from a difficult
short schedule. No live archive, model, deck, or arena identity was changed.

## Fresh live-arena slice

Sixteen genuine official-arena games were pulled from recent completed run
shards. The exact bot scored 5-11:

| Family | Games | W-L |
|---|---:|---:|
| Marnie | 6 | 1-5 |
| Rocket | 4 | 2-2 |
| Alakazam | 2 | 1-1 |
| Starmie | 2 | 0-2 |
| Crustle | 1 | 0-1 |
| Hydrapple | 1 | 1-0 |

The losses were not Wobbuffet-attack driven. Wobbuffet used `Rocket Mirror`
once and never used `Headbutt Bounce` in the loss slice. Six losses never
attacked with Mewtwo and four never established a second ready attacker, but
those are outcome associations rather than validated action corrections.

## Exact current Marnie leader panel

The authenticated bundle for current ladder contender
`AshV178MarnieFullBest` was downloaded and validated as a safe root package
(archive SHA-256
`277e7fc12a94c4ac8ce10b5b8575e3c95f2f144b049311713bb551f2db019d5e`).
Both frozen Team Rocket identities were then tested in fresh, side-swapped
official-engine games:

| Team Rocket policy | Games | W-L-D | Raw win rate |
|---|---:|---:|---:|
| ScaleBCV2 | 40 | 19-21-0 | 47.5% |
| ScaleStopV3 | 40 | 17-23-0 | 42.5% |

All 80 games ended normally with zero timeout, crash, or illegal-action
result. The panels are independent rather than paired, so the five-point gap
is not a causal estimate. They nevertheless give no reason to route the
Marnie matchup to ScaleStopV3 and show that the 1-5 live Marnie slice was much
worse than the broader direct matchup estimate.

## Decision

Continue the immutable ScaleBCV2 identity toward 300 games. It remains the
stronger of the two frozen live policies, and an exact duplicate of its policy
is currently near the top of the reset ladder. Do not deploy the inactive
Wobbuffet guard or route Marnie to ScaleStopV3 on this evidence.

## Durable evidence

- pulled shards and hashes:
  `work/arena_scalebcv2_recent_shards_20260801_3/manifest.json`;
- replay autopsy:
  `work/arena_scalebcv2_recent_autopsy_20260801_2/`;
- exact current leader bundle:
  `work/arena_bundles/AshV178MarnieFullBest_20260801.tar.gz`;
- official-engine panels:
  `work/current_leader_diagnostics/`.

Evidence boundary: pulled replays and local panels are genuine official-engine
execution. The diagnosis does not establish a counterfactual improvement or
authorize a new arena submission.

