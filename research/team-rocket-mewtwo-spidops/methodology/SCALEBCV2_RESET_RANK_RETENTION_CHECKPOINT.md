# ScaleBCV2 reset rank-retention checkpoint

Status: **live identity unchanged; rank 1 at 92/300 games**  
Observed: 2026-08-01

## Current authoritative position

The authenticated internal-arena ratings call showed
`HelbrosRocketSpidopsScaleBCV2` at rank 1 with rating 857.9 and a 62-30-0
record. Rank 2 was `AshV178MarnieFullBest` at 807.6, a 50.3-rating-point
margin. These are one identity's games; no TrainingPin or ScaleStop counts are
included.

Across the first 13 valid monitor snapshots, from 44 through 77 games, the
exact bot was rank 1 in 13/13 observations. The observed lead ranged from 13.7
to 50.4 rating points. Rank retention is therefore real so far but not treated
as guaranteed before the 300-game simultaneous check.

## Fresh official-arena replay slice

The cumulative `/data?run=latest` export timed out twice at the server's
120-second limit. `scripts/pull_recent_arena_shards.py` now uses the arena's
recent-run ledger to retrieve only small completed run shards. It validates
each tar member, stores replay payloads under short deterministic names to
avoid Windows path-length failures, preserves the original names and hashes,
and never requests or mutates a bot.

The first clean bounded pull is frozen under
`work/arena_scalebcv2_recent_shards_20260801_2/`:

- 3 exact run archives;
- 6 genuine official-arena games;
- 5 wins and 1 loss;
- 3-0 against the Marnie family;
- 2-0 in Rocket-family games;
- 0-1 against `TrainingPin - SparkStarmieMistFlatV20`.

The six-game sample is a diagnostic slice, not a standalone strength estimate.
Its useful signal is that the only loss was not an opening/setup collapse. In
that Starmie game the bot reached four Rocket Pokemon on turn 1, first attacked
on turn 3, made five attacks, reached three ready attackers and two Spidops,
and had no legal attack, attachment, evolution, or supporter stranded at the
terminal decision. It nevertheless took zero prizes while the opponent took
five. That points to matchup-level damage/prize conversion, not failure to
assemble the deck's intended board.

## Decision consequence

Do not mutate or replace the current rank-one identity. The prior N26 Starmie
attachment guard directly targeted this matchup and regressed badly in a
locked official-engine panel: 9-31 for the guard versus 17-21-2 for the
unchanged control. The fresh loss does not overturn that rejection, especially
while the immutable control is holding rank 1 with a substantial live margin.

Continue accumulating games under the unchanged identity. The completion
watcher will only seal success if one frozen bot has at least 300 of its own
rated games and fresh ratings calls both before and after exact-archive
official-engine smoke still show it at current rank 1.

## Durable evidence

- shard pull manifest:
  `work/arena_scalebcv2_recent_shards_20260801_2/manifest.json`;
- replay autopsy:
  `work/arena_scalebcv2_recent_autopsy_20260801_1/autopsy.json`;
- readable autopsy:
  `work/arena_scalebcv2_recent_autopsy_20260801_1/autopsy.md`;
- bounded puller:
  `scripts/pull_recent_arena_shards.py`;
- completion gate:
  `ARENA_300_GAME_FINALIZATION.md`.

Evidence boundary: replay rows are genuine internal-arena official-engine
execution. The diagnosis is observational and does not claim that a specific
counterfactual action would have changed the Starmie result.
