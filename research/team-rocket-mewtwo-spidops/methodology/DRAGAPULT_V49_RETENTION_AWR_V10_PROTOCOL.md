# Dragapult V49 retention-AWR V10 protocol

Frozen: 2026-08-07, before training metrics or V10 match outcomes.

## Hypothesis

The failed V9 heuristic screen showed that untouched V5 already beats the
available generic Dragapult proxy 17-7, while live V8 was only 62-121 against
Soldier PureRLV49. The remaining target is therefore policy-specific rather
than generic card-family logic. V10 asks whether a very small outcome-weighted
update on exact, on-policy V49 trajectories can learn the winning public-state
action distribution while retaining V5 as the baseline-safe blueprint.

This is same-deck policy adaptation, not hidden-hand modeling. The actor sees
only the same official observation and public belief features as V5. Opponent
name, replay outcome, hidden cards, seed, and file identity are training-only
labels and are absent from packaged execution.

## Immutable corpus boundary

- Source: 183 complete V8-versus-Soldier V49 official arena games.
- Exact V5 action audit: 14,267 decisions, 14,168 matches, 99 mismatches in 77
  games.
- Every mismatching game is excluded in full. The clean population is 106
  games: 32 wins and 74 losses, including 55 games starting second.
- Frozen split under seed 20260807: 83 train, 14 validation, 9 untouched test
  games. Exact batched V5 parity is 5,971/5,971 train, 1,106/1,106 validation,
  and 542/542 test examples.
- Parity report SHA-256:
  `3f57b075f57474104ce27588d8494b4bcff4c28e2134692bddf1f7c7367e9e41`.

## Frozen training arm

Starting checkpoint SHA-256:
`3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.

Train only the final Transformer block and action head for four sequential
epochs with batch size 128, learning rate 1e-6, weight decay 1e-4,
outcome-temperature 0.5, forward-reference KL coefficient 1.0, gradient norm
1.0, one CPU thread, and seed 20260807. Weight each game equally before
outcome advantage weighting. No checkpoint or hyperparameter may be restarted
after metrics are observed.

Epoch 1 and epoch 4 are the only match-eligible endpoints. Epoch 1 represents
the smallest safe update; epoch 4 represents the full fixed update. Offline
metrics are diagnostics and cannot authorize deployment.

## Admission gates

1. Training must preserve exact corpus hashes, split counts, feature parity,
   finite tensors, and the declared trainable tensor set.
2. Both endpoints must execute with exact packaged-runtime parity on at least
   500 official V5 observations, with every returned action legal.
3. Development runs 24 side-swapped official-engine games per arm against each
   of four frozen packages: generic Dragapult, current Marnie, Alakazam, and
   flat Starmie. Untouched V5 is the control. An endpoint must have:
   - no engine/agent/legality errors;
   - at least 12/24 in Dragapult and Marnie;
   - at least 10/24 in Alakazam and at least 8/24 in flat Starmie;
   - aggregate wins no lower than V5 minus two; and
   - no cell more than three wins below V5.
4. If both endpoints pass, select the one with more aggregate wins, then more
   starting-second wins, then the smaller epoch. Run a fresh 120-game panel
   (30 per opponent) and require at least 62/120 overall, no cell below 10/30,
   and no execution failures.
5. Only the single confirmed endpoint may be combined with V5's already
   admitted early-Spidops and exact-lethal Bangle wrapper, archived, audited,
   and submitted once to the internal arena. Do not add V8's failed Articuno
   reservation or Wobbuffet-exit guards. Do not submit to Kaggle.

Local opponents are genuine official-engine executions but not Soldier V49.
The immutable arena identity is the independent strength estimate; no local
result is a verified live-V49 performance claim.
