# N5 Rocket Blueprint Development Lock

Status: preregistered before N5 training completed and before any development game

Protocol: `rocket-greedy-compatible-ppo/v1`

## Arms

The development comparison contains exactly six arms in this order:

1. frozen initial control;
2. final checkpoint from seed 788101;
3. final checkpoint from seed 788102;
4. final checkpoint from seed 788103;
5. final checkpoint from seed 788104; and
6. an equal-weight elementwise mean of the four aligned final state dictionaries.

Only iteration 32 is eligible. Intermediate checkpoints, training win rates, and post-hoc interpolation weights cannot select an arm.

## Panel

The panel contains exactly 2,880 fresh official-engine games:

- opponents: immutable `AshMarnieGeneralV110` and `HelbrosMarnieGeneralRecoveryV4`;
- two shards per opponent;
- 120 games per arm per shard;
- 240 games per arm per opponent;
- 480 games per arm total;
- Rocket always in the second seat;
- interleaved shuffled arm schedule within each shard;
- greedy packaged-policy action path with the attack-safe END guard;
- no hidden opponent information and no exception fallback.

Locked schedule seeds:

- V110 shards: 789101 and 789102;
- Recovery V4 shards: 789201 and 789202.

## Advancement

`scripts/analyze_rocket_blueprint_n5_development.py` uses 20,000 opponent-stratified bootstrap replicates with seed 789001. At most one arm advances. It must satisfy all of the following:

- equal-opponent improvement over control is at least +3 percentage points;
- bootstrap 95% lower bound is strictly positive;
- neither opponent-specific difference is below -2 percentage points; and
- every official game completed correctly with zero fallbacks.

Eligible arms are ordered by improvement, then bootstrap lower bound, then stable arm name. Development evidence cannot authorize arena submission; it can only freeze one arm for the separate 3,000-game validation.
