# N1 Neural Plan Development Lock

Status: locked before any N1 final checkpoint or training outcome was read  
Protocol: `rocket-neural-plan-residual/v1`  
Frozen Team Rocket deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`  
Frozen blueprint SHA-256: `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`

## Arms

The development panel contains exactly six arms:

1. `zero_control`: a freshly initialized, exactly zero-output residual.
2. `seed_781101`: final iteration from training seed 781101.
3. `seed_781102`: final iteration from training seed 781102.
4. `seed_781103`: final iteration from training seed 781103.
5. `seed_781104`: final iteration from training seed 781104.
6. `mean_all4`: an elementwise equal-weight mean of the four aligned final state dictionaries.

No intermediate checkpoint, pairwise soup, post-hoc coefficient, or training win rate may enter N1 development.

## Match design

- Opponents: immutable extracted `AshMarnieGeneralV110` and `HelbrosMarnieGeneralRecoveryV4` policies.
- Seat: Team Rocket is seat 1 and must genuinely start second; the opponent is seat 0.
- Execution: complete games in the local official competition engine. The opponent is hosted through the isolated official-agent worker boundary.
- Sample: 120 games per arm per opponent, or 240 per arm and 1,440 total.
- Shards: two independently seeded 60-game-per-arm shards per opponent.
- Shard seeds:
  - V110: 782101 and 782102.
  - RecoveryV4: 782201 and 782202.
- Arms are interleaved in a seeded shuffled schedule within every shard.
- The actor receives only its ordinary observation, public belief tracker, and persistent recurrent state. No visualizer truth, opponent hidden cards, cloned branch, or future information is available.

## Development gates

An arm may advance only if all of the following hold against `zero_control`:

- equal-opponent point improvement is strictly positive;
- intervention-game rate is at least 5%;
- neither opponent-specific point difference is below -2 percentage points;
- every game is official and complete, Team Rocket started second, and exception fallbacks are zero.

Eligible arms are ordered by equal-opponent point improvement, then stratified bootstrap lower bound, then name. At most the first two advance. Development is a selection panel, not an arena-strength claim. Before locked validation, one advancing artifact and the validation seeds/sample size must be frozen in a separate lock document.

## Stop conditions

Stop N1 without test, packaging, or submission if no arm passes, if artifact provenance or hashes disagree, or if correctness fails. Do not substitute an unregistered checkpoint after seeing results.
