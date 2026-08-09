# N5 Rocket Blueprint Validation Lock

Status: preregistered before development results and before validation collection

Protocol: `rocket-greedy-compatible-ppo/v1`

## Admission

Validation is created only if exactly one of the four final N5 seeds or the aligned four-seed mean passes every development gate in `N5_ROCKET_BLUEPRINT_FINETUNE_PROTOCOL.md`. The advancing arm is the highest equal-opponent improvement among eligible arms, with bootstrap lower bound and name as deterministic tie breakers. No intermediate checkpoint is eligible.

`scripts/freeze_rocket_blueprint_n5_validation.py` copies that sole arm and the frozen control into immutable validation artifacts and records their hashes plus the development-analysis hash.

## Panel

The validation panel contains exactly 3,000 fresh official-engine games:

- arms: frozen candidate and frozen control;
- opponents: immutable `AshMarnieGeneralV110` and `HelbrosMarnieGeneralRecoveryV4`;
- two shards per opponent;
- 375 games per arm per shard;
- 750 games per arm per opponent;
- 1,500 games per arm total;
- Rocket always occupies the second seat;
- greedy packaged-policy action path with the attack-safe END guard;
- no hidden opponent information and no exception fallback.

Locked schedule seeds:

- V110 shards: 790101 and 790102;
- Recovery V4 shards: 790201 and 790202.

Each shard is marked sealed. Its match rows may be opened only after all four shards complete and only by `scripts/analyze_rocket_blueprint_n5_validation.py` with unlock token `N5_LOCKED_VALIDATION_3000`.

## Passing rule

The candidate passes only if all conditions hold:

- equal-opponent improvement over control is at least +3 percentage points;
- the opponent-stratified 95% bootstrap lower bound is strictly positive;
- neither opponent-specific difference is below -2 percentage points;
- all 3,000 games are complete official-engine executions from the second seat; and
- exception fallbacks and correctness violations are zero.

The analysis uses 50,000 bootstrap replicates with seed 790901. A pass permits broader regression and packaging. It is not an arena rank or leaderboard claim. A failure prohibits packaging, submission, intermediate-checkpoint rescue, and post-hoc validation reuse.
