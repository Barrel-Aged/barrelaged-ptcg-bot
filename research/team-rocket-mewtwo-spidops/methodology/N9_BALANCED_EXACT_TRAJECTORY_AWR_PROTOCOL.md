# N9 Balanced Exact-Trajectory Robust AWR Protocol

Status: **preregistered before trajectory collection**  
Protocol: `rocket-balanced-exact-pcgrad-awr/v1`  
Preregistered: 2026-07-30

## Motivation

N7 proved that outcome-weighted regression can move the policy materially, but
its aggregate historical Marnie data learned a V110 exploit that sacrificed
Recovery V4. N8A showed that runtime public history cannot identify those two
policies safely. N8B showed a small action-head-only change can move both
matchups in the same direction, but its fixed source delta was too weak.

N9 learns one shared, runtime-unconditioned action head from new trajectories
that are exactly balanced between the two current opponent policies. Opponent
identity is used only to balance offline gradients; it is not an actor input
and cannot leak into packaged execution.

## Frozen substrate

- Exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Exact submitted/live control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- V110 opponent model SHA-256:
  `552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b`.
- Recovery V4 opponent model SHA-256:
  `6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c`.
- Both opponents use deck SHA-256:
  `8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b`.
- Rocket is always seat 1; the official engine's first player must be seat 0.
- The exact packaged `NNAgent` greedy semantics, attack-safe single-selection
  END mask, public belief tracker, and autoregressive multi-selection semantics
  are frozen.

## Fresh trajectory collection

Collect exactly 1,200 complete strict official-engine games: 600 against each
opponent. Four immutable shards contain 300 games each:

- V110 seeds 797101 and 797102;
- Recovery V4 seeds 797201 and 797202.

The focal actor is the frozen control and is recorded as `N9RocketControl`.
Each game is a standalone arena-compatible JSONL replay containing both public
observation streams, chosen legal actions, player/deck metadata, and terminal
winner. Files are written atomically. Any engine error, timeout, illegal
action, wrong seat marker, unexpected deck handshake, or strict-actor
exception fails its shard. Draws remain in the immutable corpus but are
excluded from binary AWR training.

No replay from N1-N8 or the live arena enters N9 training.

## Exact behavior audit

Before feature extraction, replay every recorded focal decision through a new
`NNAgent` instance initialized from the frozen model and deck. Forced full-set
ordering differences may be counted as equivalent; every genuine mismatch
causes the entire game to be excluded. Because the collector uses the same
strict greedy path, the expected result is zero excluded games. The parity
report and replay-tree hashes must be frozen in a training execution lock
before optimization.

## Split and weights

Split games independently within each exact opponent stratum after sorting by
immutable filename. Assignment repeats every ten games:

- indices 0-7: training (80%);
- index 8: validation (10%);
- index 9: test (10%).

No game crosses splits. With 600 decisive games per opponent this yields 480
train, 60 validation, and 60 test games per opponent, subject only to draws or
whole-game parity exclusions.

Every game contributes equal base mass regardless of length. For each exact
opponent, estimate the training win rate with the same five-game empirical
Bayes shrinkage used by N7. An action from a game with outcome `y` receives
weight proportional to `exp((y - group_rate) / 0.5)`. Normalize weights within
each opponent so V110 and Recovery contribute equal total mass.

## Robust shared-head optimization

Only `logit_head.weight` and `logit_head.bias` are trainable. All embeddings,
feature projections, Transformer layers, auxiliary/value heads, and belief
machinery remain byte-frozen.

Fixed optimization:

- seed 797001 and one CPU thread;
- four epochs;
- 64 examples per opponent in every paired minibatch;
- AdamW, learning rate 0.000003, weight decay 0.0001;
- frozen-reference KL coefficient 0.25;
- gradient norm cap 1.0.

Compute one advantage-weighted imitation gradient per opponent. If their dot
product is negative, apply symmetric PCGrad projection before averaging them.
Then add the frozen-reference KL gradient. This prevents a directly opposing
gradient from one policy overwhelming the other while still learning a single
shared action head. PCGrad conflict rates and cosine similarities are
diagnostic only and cannot select a checkpoint.

Save exactly `epoch_01` through `epoch_04`. Offline validation/test metrics do
not choose an epoch; all four immutable epochs enter the same official-engine
development panel with control.

## Development and advancement

After exact packaged-runtime parity for all five arms, run the same balanced
2,400-game panel geometry used by N7/N8B: 240 games per arm/opponent cell,
Rocket always second, fresh locked seeds to be frozen before execution.

At most one epoch may advance. Required gates:

- strict correctness and zero fallbacks;
- at least +2.0 percentage points equally stratified improvement;
- strictly positive worst-opponent point difference;
- 95% stratified bootstrap lower bound greater than -0.5 points; and
- 480 games per arm.

Tie-break by stratified improvement, then worst-opponent difference, then
earlier epoch. A pass is development selection only.

## Post-development requirements

Any selected epoch still requires at least 3,000 fresh sealed validation games,
broad Rocket/Alakazam/Soldier regression, first-seat preservation or an exact
public seat gate, packaged archive parity and deck audit, then a live arena
submission and at least 300 rated games. No N9 artifact may be submitted merely
because its offline loss improves.
