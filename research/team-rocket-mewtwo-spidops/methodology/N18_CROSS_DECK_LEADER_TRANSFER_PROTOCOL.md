# N18 Cross-Deck Leader Transfer Protocol

Status: **preregistered before fresh match collection**  
Protocol: `rocket-n18-cross-deck-leader-transfer/v1`

## Motivation

The live exact-deck Team Rocket bot remains
`HelbrosRocketSpidopsScaleBCV2`, model SHA-256
`3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
N1-N17 did not admit a replacement.  N17 closes the old localized N6R
residual family; N18 must not resample, enlarge, checkpoint-select, or otherwise
retune that rejected evidence.

The N18 evidence audit instead found genuinely new public official-engine
evidence.  Current rank one and rank two are aliases of one strategy with
byte-identical model and history residual.  Their deduplicated corpus contains
984 games, and rank one is 117-61 in 178 direct games against the exact live
Rocket bot.  The leader and live neural architectures are identical, with only
the same 1,184,257 last-block/action-head parameters differing.  This supports
a closed cross-deck transfer test, but the leader deck is not Team Rocket and
no transfer-strength claim is made before fresh games.

## Frozen evidence and source artifacts

- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- evidence audit JSON SHA-256:
  `f0441e0e602421c0ccbb57447e9b5857af442c93e7b6a02224cd4f09c24e9a0c`;
- evidence audit Markdown SHA-256:
  `f6b4d74c007d3dd5cb438eecdd4a1560aef64b311d0fe5bcbb1b9c4a177ee084`;
- live source tree SHA-256:
  `11988ad1e5ab0b81d993ad54bf32b1a903caf89b6a4040aacf57e0f1d6a8480a`;
- leader source tree SHA-256:
  `1a78564bb3304fc4c968e14cb01245eba2c232fe32594c0cfdb563a01a76e828`;
- leader model SHA-256:
  `6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c`;
- leader history residual SHA-256:
  `5df9bd6e9f2d733092b0653765b4d1d3f082a488cfe68fe31f23c94f3aa43f4f`;
- arm builder SHA-256:
  `cf61aad5b38583165b69243d3c6b347a1a08225e0ec8c11a9ea35a4c8853baaa`;
- arm manifest SHA-256:
  `57ab74eaaf05aa2f4e36a09f1555aef4eadf1d84dd0415c59376de036bfbed18`;
- packaged-runtime audit SHA-256:
  `93c2e64764108ee78415b13ce3de3bf6e0c7bc8256b44804a002081e802f0b93`;
- packaged-runtime auditor SHA-256:
  `4163f712696045f6894c0fb434f8b1f2df0c89616e6533fc8ff92b28d4891a78`.

## Closed arm family

All arms return the exact required 60-card Team Rocket deck.  No arm may be
trained, edited, rescaled, or checkpoint-selected after this lock.

1. `control`: exact live model and live runtime; bundle tree SHA-256
   `6fcbf88b34b40024cdd6a67efd5d07071d8307db117e0f8b6f95a3a93f00c983`.
2. `live_residual`: live model plus the frozen leader history residual under
   the leader runtime; bundle tree SHA-256
   `d6f432e67c4152b441b92943adec97b253666394a36bac2b6a6522709f3105ea`.
3. `leader_base`: frozen leader model under the leader runtime without the
   history residual; bundle tree SHA-256
   `1da4f030f276c961fcfc720619cfa98b932f61a68b39312d594a3fed1c0e9fb1`.
4. `leader_residual`: frozen leader model and frozen leader history residual;
   bundle tree SHA-256
   `0b4114cb54d77e6e6d07b3d5e666f2c6930df2cda12879820345524f86eb140c`.

`bridge_base`, tree SHA-256
`8fabf733dbf83eb4ac168930c73f51677ea3ec557746cd1a84ee008a037c2e08`,
is diagnostic only.  It places the live model under the leader runtime without
the residual and matched the same-run live control on 781/781 frozen packaged
decisions.  It receives no match allocation and cannot advance.

The runtime audit observed 781/781 legal actions for every packaged arm.  On
the same observations, selected-option-set disagreement from control was
80/781 for `live_residual`, 247/781 for `leader_base`, and 218/781 for
`leader_residual`.  These disagreements establish distinct policies only;
they are not counterfactual values or strength evidence.  The live rerun also
differed from two archived decisions, consistent with the restart-level
nonreproducibility already documented by N10.  Archived byte reproduction is
therefore not an N18 strength or parity gate.

## Development opponents and official-engine boundary

The locked development panel uses three frozen opponent bundles:

- current rank-one `AshMarnieGeneralV177MultiFailure`, tree SHA-256
  `1a78564bb3304fc4c968e14cb01245eba2c232fe32594c0cfdb563a01a76e828`;
- `AshMarnieGeneralV110_extracted`, tree SHA-256
  `3bb5e40bb1583e31470eee263ed615c9207427c7b5d2199e80fbe3571e2cbb2b`;
- `HelbrosMarnieGeneralRecoveryV4_extracted`, tree SHA-256
  `caf361fd297ab63b24338650b6e396a296b961bb6ba61d2519799a90dff738f4`.

Fresh matches are genuine execution by the bundled native `cg` official game
engine.  The engine is imported only from the frozen `control` bundle so
candidate runtime differences cannot change the match engine.  Every arm and
opponent is executed through the arena evaluator's isolated packaged-agent
worker, including setup deck handshake, persistent within-game state, attack
guard, and history residual where present.  The collector verifies every
returned action against the official selection before applying it.

The cross-deck hypothesis, weight transfer, residual transfer, confidence
interval, and candidate-selection rules are research abstractions.  They are
not features of the official engine.

## Locked development schedule

The panel is equally stratified by three opponents and both Rocket seats.  Six
shards are collected.  Each shard contains exactly 120 games for each of the
four match-tested arms, or 480 rows:

| Seed | Opponent | Rocket seat | Output |
|---:|---|---:|---|
| 818101 | current rank one | 0 | `leader_seat0.jsonl` |
| 818102 | current rank one | 1 | `leader_seat1.jsonl` |
| 818201 | V110 | 0 | `v110_seat0.jsonl` |
| 818202 | V110 | 1 | `v110_seat1.jsonl` |
| 818301 | RecoveryV4 | 0 | `recovery_seat0.jsonl` |
| 818302 | RecoveryV4 | 1 | `recovery_seat1.jsonl` |

Total development size is exactly 2,880 official-engine games: 720 per arm,
240 per opponent per arm, and 360 per Rocket seat per arm.  Within each shard,
the four-arm schedule is shuffled once by the locked seed.  First player is
official-engine player zero, so Rocket seat zero and one are separately
identified in every row.

Before collection, an execution lock must freeze the exact collector,
analyzer, evaluator, worker helper, canonical engine bundle, opponent
fingerprints, arm manifest, all output paths, and the nonexistence of the panel
outputs.

During collection, no row contents, wins, losses, partial manifests, rating
proxies, or action telemetry may be inspected.  Only process health, file
existence, and row counts may be observed until all six shards and sidecars
seal.  An interruption must preserve completed material and receive a durable
recovery lock before any rerun.

## Locked analysis and multiplicity control

The analyzer uses 20,000 stratified bootstrap replicates under seed `818001`.
Each replicate resamples games within each of the six opponent-by-seat cells
and gives every cell equal weight.  Each challenger is compared with the
shared control.

Three challenger comparisons are a closed family.  The report includes
ordinary 95% intervals and Bonferroni-simultaneous 95% intervals, using the
two-sided quantiles 0.0083333333 and 0.9916666667 for each challenger.

A challenger passes development only if all of the following are true:

1. every correctness, provenance, count, legality, engine, deck, and
   no-fallback gate passes;
2. equally stratified aggregate win-rate improvement is at least +2.0
   percentage points;
3. improvement is strictly positive for each of the three opponent aggregates;
4. improvement is strictly positive for each Rocket-seat aggregate;
5. no individual opponent-by-seat cell is worse than control by 5.0 percentage
   points or more;
6. the Bonferroni-simultaneous lower bound is greater than -0.5 percentage
   points.

If multiple challengers pass, the sole development winner is the one with the
largest equally stratified aggregate improvement.  An exact numerical tie is
resolved by the fixed order `leader_residual`, `leader_base`,
`live_residual`.  No nonpassing arm may advance, even if it has the largest
raw win rate.  The bridge arm cannot advance.

## Fresh confirmation if and only if development passes

A selected development winner receives no deployment authority.  It may only
enter a fresh 3,200-game confirmation against the current-rank-one and V110
bundles, both Rocket seats, with new seeds and 400 games per arm in each of the
four opponent-by-seat cells.  The confirmation compares only control and the
selected arm.

Confirmation requires strict correctness, at least +2.0 percentage points
aggregate improvement, positive improvement for each opponent and each seat,
no cell below -2.0 percentage points, and a strictly positive stratified 95%
bootstrap lower bound.  Passing confirmation permits only broad both-seat
regression against additional frozen archetypes and packaging audit.

Submission is permitted only after confirmation and broad regression both
pass.  The existing live bot remains unchanged until then.  No Kaggle or
internal-arena submission, #1 claim, superhuman claim, or verified performance
claim is authorized by this protocol alone.
