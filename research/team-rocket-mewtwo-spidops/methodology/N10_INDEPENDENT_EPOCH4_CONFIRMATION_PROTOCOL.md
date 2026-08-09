# N10 Independent Epoch-4 Confirmation Protocol

Status: **preregistered before arms are built or confirmation games collected**  
Protocol: `rocket-n10-independent-epoch4-confirmation/v1`  
Preregistered: 2026-07-30

## Question

N9 epoch 4 improved the equal-opponent point estimate by 5.42 percentage
points and was positive against both frozen Marnie policies, but its bootstrap
lower bound was -0.83 points rather than the locked requirement above -0.50.
N9 therefore failed and remains non-deployable.

N10 asks one new question on entirely untouched games: does the single,
already-selected epoch-4 checkpoint reproduce a positive, practically useful
gain over the live control? There is no checkpoint selection, hyperparameter
selection, or intermediate rescue in N10.

## Frozen candidate and substrate

- Exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Live control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Fixed N9 epoch-4 model SHA-256:
  `bdafc1ce36f7e572554930a1a9b1beff541a351ece09f934d0416b4ce269a020`.
- V110 model SHA-256:
  `552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b`.
- Recovery V4 model SHA-256:
  `6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c`.
- Both opponents use deck SHA-256:
  `8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b`.
- Existing packaged-runtime parity is frozen at 461/461 exact decisions for
  epoch 4 and 389/389 for control, with zero mismatches.

Epoch 4 differs from control only in `logit_head.weight` and
`logit_head.bias`; all other policy tensors, the official adapter, public
belief tracker, legal-action handling, deck, and greedy/autoregressive runtime
semantics remain frozen. The actor cannot observe opponent identity or hidden
opponent cards.

## Independent official-engine panel

Collect exactly 3,200 fresh games: 800 games per arm/opponent cell for the two
arms (`control`, `candidate`) and two opponents. Rocket is always seat 1 and
the official engine's first player is seat 0.

Four independent shards contain 400 games per arm:

- V110 seeds 799101 and 799102;
- Recovery V4 seeds 799201 and 799202.

Each shard randomizes its two-arm schedule before play. It must use strict
`play_game` execution with no exception fallback. An engine error, timeout,
illegal action, wrong deck/model hash, wrong seat, or incomplete shard fails
the experiment. Rows and sidecar manifests are append-visible for operational
monitoring, but win counts and arm comparisons must not be inspected until all
four 800-row shards and their manifests are complete.

N10 may not use N9 development games, live-arena outcomes, or any earlier
recovery corpus in its estimate.

## Locked analysis and pass gate

The analyzer uses 50,000 stratified bootstrap replicates with seed 799001. It
compares candidate and control separately within each opponent and gives both
opponents equal weight. N10 passes only if all conditions hold:

- all 3,200 rows pass strict correctness checks with zero fallbacks;
- exactly 1,600 games exist for each arm and 800 for every arm/opponent cell;
- equal-opponent candidate improvement is at least +2.0 percentage points;
- the two-sided 95% stratified-bootstrap lower bound is strictly above zero;
- the candidate point estimate is strictly better than control against each
  opponent.

There is one fixed candidate and no tie-break. Any failed condition rejects
N10. Partial-sample stopping, sample-size extension, seed replacement,
checkpoint substitution, and post-hoc subgroup rescue are forbidden.

## Authority boundary

A pass confirms one narrow second-seat result; it does not authorize immediate
submission. It admits epoch 4 to a separate broad regression covering both
seats and the frozen Rocket, Alakazam, and Soldier opponent families, followed
by archive construction, exact packaged-runtime/deck audit, and an independent
submission decision. Only a qualifying immutable live bot with at least 300
rated games and observed rank one completes the project goal.

