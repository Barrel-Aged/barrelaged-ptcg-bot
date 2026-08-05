# Arena 300-Game Rank-One Monitor

Status: **active read-only monitoring; goal not yet met**  
Started: 2026-08-01

The arena reset on 2026-08-01. The exact qualifying Team Rocket candidate
`HelbrosRocketSpidopsScaleStopV3` initially appeared at rank 1, but with only
one rated game. That was provisional and did not satisfy the user goal. During
the same reset, immutable `HelbrosRocketSpidopsScaleBCV2` emerged as the
stronger qualifying entry, so the monitor now follows both exact-deck
identities and never combines their game counts.

Frozen identities:

- bots: `HelbrosRocketSpidopsScaleBCV2` and
  `HelbrosRocketSpidopsScaleStopV3`;
- deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- ScaleBCV2 model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- ScaleStopV3 model SHA-256:
  `c6b808e3e50b5fffcfd44b59ad2a541c6661e8d22915b1cbb5ade5a57464d8f2`;
- success condition: one immutable identity is current arena rank exactly 1
  after at least 300 of its own rated games.

`scripts/monitor_arena_rank_goal.py` reads arena ratings only, verifies both
local deck/model pairs on every poll, and writes atomic snapshots under
`work/arena_rank_goal_monitor/`. It stops automatically only when the exact
success condition is observed. A rank-one position before 300 games is not
reported as completion.

The arena may temporarily omit a bot while recomputing a newly reset rating
graph. The monitor records per-bot presence and continues; absence, a request
failure, a stale snapshot, or adding games from two identities can never
satisfy the goal.

No Kaggle action, arena submission, rating mutation, or bot replacement is
performed by the monitor.

Final completion is intentionally a separate fail-closed operation. See
`ARENA_300_GAME_FINALIZATION.md` and
`scripts/finalize_arena_rank_goal.py`; the monitor snapshot alone is never a
completion certificate.
