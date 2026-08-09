# N8B Recovery Logit-Delta Transfer Protocol

Status: **preregistered before arm construction or match execution**  
Protocol: `rocket-recovery-logit-delta-transfer/v1`  
Preregistered: 2026-07-30

## Hypothesis

`AshMarnieGeneralV110` and the rank-leading
`HelbrosMarnieGeneralRecoveryV4` use the exact same 60-card deck, model
configuration, embeddings, feature projections, Transformer, auxiliary head,
and value head. Their model files differ in exactly two tensors:
`logit_head.weight` and `logit_head.bias`. The L2 norm of the weight delta is
0.03266454 and the absolute bias delta is 0.000057416.

This is an unusually clean strategic-policy contrast. N8B tests whether adding
that small Recovery-minus-V110 action-head delta to the Rocket-adapted
blueprint transfers broadly useful action preferences without undoing Rocket's
deck-specific representation.

## Frozen provenance

- Exact Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Development control model SHA-256:
  `cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff`.
- Submitted/live-equivalent action model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- V110 source model SHA-256:
  `552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b`.
- Recovery V4 source model SHA-256:
  `6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c`.
- Shared Marnie deck SHA-256:
  `8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b`.

Before use, the builder must prove that the two source models have identical
configuration and state-dict keys/shapes, and that every tensor except the two
action-head tensors is byte-identical. It must also prove that the development
control and submitted model differ only in `value.*` tensors, so their greedy
action policies are identical.

## Fixed arms

Let `D = RecoveryV4.logit_head - V110.logit_head`. Construct exactly five
development arms from the immutable Rocket control:

1. `control`: alpha 0.0;
2. `delta_050`: alpha 0.5;
3. `delta_100`: alpha 1.0;
4. `delta_150`: alpha 1.5;
5. `delta_200`: alpha 2.0.

For each candidate, only `logit_head.weight` and `logit_head.bias` may differ
from control, and the new tensors must equal `control + alpha * D` within exact
float32 construction semantics. There is no fitting, trajectory reuse,
checkpoint selection, search, hidden information, or reward optimization.

## Runtime and correctness gate

Every arm must load through the exact `soldier.net.load_policy` path, produce
finite tensors, retain the frozen deck, and match the strict official-engine
greedy actor on every audited decision. The attack-safe END rule applies to
single selections; autoregressive multi-selection semantics remain unchanged.
Any illegal action, engine error, timeout fallback, agent exception, model-load
failure, unexpected changed tensor, or parity mismatch rejects that arm.

## Locked development panel

The Rocket policy always occupies seat 1, the empirically weak seat. Opponents
are the immutable extracted bundles:

- `AshMarnieGeneralV110_extracted`;
- `HelbrosMarnieGeneralRecoveryV4_extracted`.

Run two independent interleaved shards per opponent, with 120 games per arm in
each shard. Locked seeds are:

- V110: 796101 and 796102;
- Recovery V4: 796201 and 796202.

The complete panel is 240 games per arm/opponent cell, 480 games per arm, and
2,400 official-engine games total. All five arms remain in every shard; there
is no stage-one arm dropping. Results may not be inspected to change arms,
seeds, counts, opponents, or gates.

## Advancement rule

Control win rates are computed separately by opponent. Candidate differences
are equally weighted across the two opponents. A stratified, game-level
bootstrap uses 20,000 replicates and seed 796001.

At most one candidate advances. It must satisfy all of:

- strict correctness across every match;
- stratified improvement over control of at least +2.0 percentage points;
- worst-opponent point difference strictly greater than -1.0 point;
- 95% bootstrap lower bound greater than -2.0 points; and
- at least 480 completed games.

If multiple candidates pass, select the largest stratified improvement, then
the largest worst-opponent difference, then the smaller alpha. These are
development-selection gates only, not deployment authority.

## Required work after any development pass

A selected arm must still pass:

1. independent sealed validation with at least 3,000 fresh second-seat games
   against both Marnie policies;
2. broad regression against Rocket, Alakazam, and Soldier opponents;
3. first-seat preservation or an audited public seat-gated baseline router;
4. exact packaged-runtime parity, deck/hash audit, legality smoke, telemetry,
   and replay capture; and
5. live arena submission and at least 300 rated games.

No rank, rating, or superhuman claim is permitted before live evidence. If no
arm clears development, N8B ends with no validation, packaging, or submission.

