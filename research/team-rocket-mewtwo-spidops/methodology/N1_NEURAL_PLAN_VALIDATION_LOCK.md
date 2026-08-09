# N1 Neural Plan Validation Lock

Status: locked before any validation game was collected  
Protocol: `rocket-neural-plan-residual/v1`

## Frozen candidate

- Candidate: `seed_781104`, final iteration 16 only.
- Candidate residual SHA-256: `550457082afefea3426d429c0d303b4d0441a3f3642758f2cf47a1b6c31a0a3e`.
- Control: `zero_control`.
- Control residual SHA-256: `02e9b23b75dfaef085b1f9ad782a75cdb3a3b1ed5969d164c10075ab1bc35224`.
- Frozen Team Rocket deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Frozen tactical blueprint SHA-256: `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`.
- Validation arm manifest SHA-256: `2f9051498bfb86e4f9e77b00e733347247ca6f996a8c053e20a92dd7fbb0bc95`.

No learned weight, residual scale, action threshold, checkpoint, deck card, or runtime behavior may change during validation.

## Why this candidate

The locked N1 development ranking selected `seed_781104` first. Across 240 fresh development games it scored 43.33% versus 38.75% for the interleaved control, an equal-opponent difference of +4.58 percentage points. Its opponent differences were -0.83 points against `AshMarnieGeneralV110` and +10.0 points against `HelbrosMarnieGeneralRecoveryV4`; intervention-game rate was 95.42%; exception fallbacks were zero. The development bootstrap interval was wide and included zero, so this is a validation candidate rather than an established improvement.

Development analysis SHA-256: `5979b5a6d56d6a141e2928ba021d5914382f90777cb88089ddab07ca62fd46ae`.

## Frozen execution code

- `scripts/neural_plan_residual_runtime.py`: `306e7b42b7f1261653ed7b3557b61a5ea60b9f0751a83a4d0ea9cb073acedd54`.
- `scripts/collect_neural_plan_panel.py`: `2a2c7cae843fadf06228a4f710c100e76ef92c1f35207a25121871521851a2b0`.
- `scripts/analyze_neural_plan_panel.py`: `688797dec07f4a0539cf0f26e6260afd921c4c80ebe74507a826ce40eb87209e`.

The official engine executes every complete game. The recurrent plan residual and value heads are research abstractions; they are not official primitives or a game-theoretic certificate. The actor receives ordinary redacted observations only.

## Validation design

- Team Rocket is always seat 1 and must genuinely start second.
- Opponents are the immutable extracted policies:
  - `AshMarnieGeneralV110_extracted`, model SHA-256 `552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b`.
  - `HelbrosMarnieGeneralRecoveryV4_extracted`, model SHA-256 `6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c`.
- Both opponent decks have SHA-256 `8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b`.
- Candidate and control are interleaved in a seeded shuffled schedule within every shard.
- Two shards per opponent, 250 games per arm per shard.
- Totals: 500 candidate and 500 control games per opponent; 2,000 games overall.
- Shard seeds:
  - V110: 784101 and 784102.
  - RecoveryV4: 784201 and 784202.
- Analysis: opponent-stratified nonparametric bootstrap, 50,000 replicates, seed 784901.

Validation shards must not be inspected until all four are complete.

## Pass gates

Validation passes only if all conditions hold:

1. Equal-opponent candidate-minus-control point improvement is at least +3 percentage points.
2. The stratified 95% bootstrap lower bound is strictly greater than zero.
3. Neither opponent-specific point difference is below -2 percentage points.
4. Every game is a complete official-engine result with Team Rocket starting second.
5. Exception fallbacks, illegal actions, engine/agent correctness failures, and hidden-information violations are all zero.

If any gate fails, N1 stops without test, packaging, or arena submission. A pass authorizes—but does not itself complete—an untouched test, broader regression checks, packaging audit, and arena submission decision.
