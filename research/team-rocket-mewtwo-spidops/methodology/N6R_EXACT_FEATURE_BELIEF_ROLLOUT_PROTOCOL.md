# N6R Exact-Feature Belief-Rollout Protocol

Status: preregistered corrective development protocol. Fixed before any
full-size N6R shard is collected. It does not authorize validation, packaging,
or arena submission.

Execution note: the original frozen trainer failed closed before creating a
student. The deployment-equivalent, lossless repair and its replacement hash
are frozen in `N6R_DISTILLATION_PARITY_REPAIR.md`; that amendment governs N6R
distillation while leaving this protocol's data, arms, and strength gates
unchanged.

## Why N6R exists

N6's paired official-engine rollout values were complete and independently
replicated, but its offline policy feature reconstruction was not lossless:
9/400 training roots failed to reproduce the collector's frozen baseline
action. N6R corrects only that data boundary. It does not alter the hidden
allocation prior, candidate search, terminal utility, estimator, student
objective, or development gates.

The complete N6 result and invalidation are recorded in
`N6_BELIEF_ROLLOUT_RESULT.md`. N6 files and the small N6R smoke are excluded
from N6R training, selection, validation, and strength claims.

## Frozen substrate

- Deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Initial policy SHA-256:
  `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`.
- Initial policy directory:
  `artifacts/team_rocket_recovery/candidates/scale_value_calibrated_v1`.
- Opponents: frozen `AshMarnieGeneralV110_extracted` and
  `HelbrosMarnieGeneralRecoveryV4_extracted` bundles.
- Seat: Rocket starts second.

## Exact actor-feature boundary

At each retained root, after the frozen greedy actor chooses its baseline, the
collector deep-copies the actor's persistent belief state and recreates the
feature tensors from the same normal observation. It serializes every tensor
with explicit dtype, shape, values, and a canonical SHA-256 digest. The
snapshot may contain the actor's own visible/private cards and derived public
belief features because those are legal actor inputs. It may not contain the
opponent's true hand, realized hidden allocation, visualizer state, or raw
sampled particle identities.

Before any descendant rollout, the collector feeds the captured snapshot back
through the byte-frozen model, applies the same attack/end guard, and requires
the exact baseline action. A mismatch aborts the shard; dropping or repairing
the row is forbidden. Offline training deserializes this snapshot rather than
reconstructing belief from a fresh tracker and independently requires the same
baseline parity for every row.

## Teacher and collection

The N6 teacher is otherwise unchanged:

- at most two distinct early-turn roots per accepted game, turns 1 through 8;
- blueprint plus two semantically distinct alternatives;
- 12 public-consistent hidden allocations and two common-random coin streams;
- 24 complete matched comparison sets per root and three terminal action
  continuations per set;
- frozen greedy neural continuations for both players;
- genuine official `cg` engine execution through terminal state;
- maximum 600 rollout decisions and 600 seconds per root, fail closed;
- no leaf-value substitution and no online search in the student.

Training is four 25-game shards per opponent:

- V110 seeds: `792111`, `792112`, `792113`, `792114`;
- Recovery seeds: `792211`, `792212`, `792213`, `792214`.

Independent selection is three 20-game shards per opponent:

- V110 seeds: `792121`, `792122`, `792123`;
- Recovery seeds: `792221`, `792222`, `792223`.

The eight training shards must finish and seal before the six selection shards
start. Outcomes remain unopened until every shard in that split has a valid
manifest. Every manifest must bind the output hash, exact arguments, 100%
feature parity, 100% root/pair completion, zero errors, official descendant
execution, no realized-truth read, and no raw hidden allocations.

## Estimation and student

The clustered estimator and hard-change rule remain identical to N6. A
non-baseline hard teacher action requires at least 22 matched sets, both
six-allocation halves nonnegative against baseline, and a strictly positive
allocation-cluster bootstrap 95% lower bound. Soft probabilities redistribute
only the frozen shortlist mass by terminal-mean softmax with coefficient
`clip(range, 0, 1) * min(1, sqrt(clusters / 12))` and a frozen-reference KL
penalty.

Only the final relational transformer block and action head train. Hyperparameters
are fixed at four epochs, batch size 64, AdamW learning rate `2e-5`, weight
decay `1e-4`, seed `792301`, and 2,000 clustered bootstrap replicates per root
during encoding. The residual cross-entropy and `1e-4` KL dead zone are
unchanged. Only the final epoch is eligible.

Arms are unchanged:

- A: temperature `0.35`, KL `1.0`;
- B: temperature `0.50`, KL `1.0`;
- C: temperature `0.75`, KL `1.0`;
- D: temperature `0.50`, KL `2.0`;
- control: byte-frozen initial policy.

The locked descriptive analyzer uses 10,000 clustered bootstrap replicates and
seed `792401`.

## Development strength panel

Any four valid students plus control enter four independently scheduled,
interleaved official-engine shards. Each shard gives every arm 120 games,
always from the second seat:

- V110 seeds `793101`, `793102`;
- Recovery seeds `793201`, `793202`.

Thus every arm receives 480 games: 240 per opponent. Analysis uses 20,000
opponent-stratified independent-game bootstrap replicates with seed `793001`.
An arm advances only if it has no legality/parity failure, improves aggregate
paired win rate by at least 1.5 percentage points, has a strictly positive
worst-opponent point estimate, and has a 95% lower bound above -0.5 percentage
points. Exactly one highest-ranked passing arm or no arm advances.

First-seat deployment remains byte-frozen control; an advancing N6R model may
serve only the public `firstPlayer`-conditioned second-seat branch. Independent
validation, regression, packaging, and arena submission require a later
immutable lock.

## Frozen implementation hashes

- N6R collector:
  `af22516fbd4a8c2bcd0c03c7edbe9822cb8ff0a8d4c391352fee16fabe3115ec`;
- inherited N6 rollout runtime:
  `fc1fab97dbceee170182417df93eecfd11059747020f9ce70f1e29ce681c065b`;
- inherited official causal helper:
  `8309391b47fc39bd869d1c4d31e2af51e2f46af04d1305cc8c891d1cde85a693`;
- typed feature snapshot codec:
  `4b76b88729621bea1b2c621283e6616de56f11a5810a803e22d2623cc2972906`;
- clustered estimator:
  `08a87e2b8e1a94a6056d5388ed63c9a6e583aecc7355574d794473d6f420c1d4`;
- trainer:
  `470f95e0bb9fabe3d6c794cd607a3a4f2878f39d3ca0ff6e57ddd6d348e988a3`;
- analyzer:
  `82f2a6b7fc790b3f0ccd49c5e8f88647778678a4ec95672f1bc8c705d7e3706f`;
- snapshot tests:
  `f78bd061178cb4d388f87e565a1f1eaf31150de0598339bb5e46ff32163dddd6`;
- distillation guard tests:
  `cb68946d2ee733b6713b671a86cfb8d9167bd61cd66e6eff2291a20caa06b829`.

After the panel specification above was frozen, its direct implementation was
completed while N6R training collection was still running and before any N6R
outcome or student existed. No seed, count, statistic, or gate changed:

- arm manifest builder:
  `f3ba654720988a71b86adf22b5f4b2e16076f64f5064c88aa36093f9d494a995`;
- official development-panel collector:
  `a49626e614d2534ddd4343780b00e807dcaa8549eaa7a78f6061dd83ce068955`;
- locked panel analyzer:
  `a663a3eb0b2f95762012e4d561ebb5b55bc976c3ed0aafa9253a0c136c203ed1`.
