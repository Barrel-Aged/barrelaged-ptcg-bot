# N21 Top-Action-Margin Leader BC Protocol

Status: **preregistered before training**  
Protocol: `rocket-n21-top-action-margin-bc/v1`

## Motivation

N19 proved that fresh winning leader actions are learnable but its KL-constrained
update changed about 3% of exact-Rocket top choices.  N20 proved that no global
source-to-terminal scalar on the fixed grid simultaneously passed the teacher
and 99.5% retention gates.  Both families are closed.

N21 starts again from the exact live model and changes the constraint itself.
Rather than treating a small average KL as sufficient, N21 directly protects
the live winning action and a capped portion of its logit margin on fresh
Rocket, N6, and N9 states during every optimizer step.

This is one new supervised training run.  It is not interpolation, checkpoint
selection, reward optimization, search, or hidden-information modeling.

## Frozen sources

- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- N19 corpus audit SHA-256:
  `d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed`;
- N19 result SHA-256:
  `cfb2560fca269d969a9490453db28b7f7a3bf6598e96864cfd886ecc1962f227`;
- N20 projection report SHA-256:
  `83f8094ecfacbeb6c73caaf899af2fb7cbbfc9779ce5f4ec9d77bb23218319e0`;
- N20 result SHA-256:
  `84fa85599e38cb50dc03265e8d090666467993ee34681dda62b46f82861c1d8c`;
- N17 replicated-target audit SHA-256:
  `c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c`.

## Training support

N21 uses the unchanged N19 splits:

- 9,483 fresh-leader teacher train targets;
- 6,240 disjoint fresh exact-Rocket retention train targets;
- the frozen teacher and retention validation/test splits.

It also uses:

- the 469 N6 examples not belonging to any of the 170 strict or replicated N17
  target states;
- all 6,753 N9 exact-Rocket held-out trajectory examples, retaining both frozen
  opponent groups.

For every retention example, the exact live model supplies legal-action
probabilities, top action, and top-versus-runner-up margin.  The protected target
margin is `min(base_margin, 0.25)`.  Candidate margin shortfall is penalized by
`softplus((target_margin - candidate_margin) / 0.10) * 0.10`.

Full relational forward passes are mandatory for N6/N9 because the complete
final transformer block is trainable.  Cached norm2-only approximations are
forbidden.

## Sole training run

- initialization: exact live model, never N19 or N20;
- trainable tensors: final transformer block plus action head, exactly 14
  tensors and 1,184,257 parameters;
- epochs: exactly 2;
- teacher batch size: 96;
- fresh-retention batch size: 96;
- N6-anchor batch size: 96, deterministically cycled;
- N9-anchor batch size: 96, deterministically cycled;
- steps: exactly 198;
- optimizer: AdamW;
- learning rate: `2e-6`;
- weight decay: `1e-4`;
- label smoothing: `0.01` over legal actions;
- gradient norm cap: `1.0`;
- seed: `821001`;
- threads: 6;
- checkpoint rule: terminal epoch 2 only.

The locked loss is:

`teacher_CE`
`+ 0.25 * teacher_KL_to_live`
`+ 4.0 * fresh_margin_loss`
`+ 1.0 * fresh_KL_to_live`
`+ 1.0 * N6_margin_loss`
`+ 0.25 * N6_KL_to_live`
`+ 1.0 * N9_margin_loss`
`+ 0.25 * N9_KL_to_live`
`+ 0.01 * source_parameter_anchor`.

Teacher, fresh retention, N6, and N9 orders are independently shuffled once per
epoch using fixed seed offsets and cycled without outcome feedback.  No game
reward, official-match result, private opponent card, N18 result, N19
checkpoint, N20 alpha, or search value enters the gradient.

## Terminal offline gates

The epoch-2 model passes only if every gate is true:

1. exact deck, model configuration, keys, shapes, finite values, and
   serialization are valid;
2. only the 14 preregistered tensors differ from live;
3. teacher validation top-1 improves by at least +2.0 percentage points and
   validation loss decreases;
4. teacher test top-1 improves by at least +1.0 percentage point and test loss
   decreases;
5. live top-action agreement is at least 99.5% on each fresh retention split;
6. fresh retention mean KL is at most 0.01 on each split;
7. recorded-action retention validation and test are each no more than 0.5
   percentage points below live;
8. N6 unchanged-anchor top-1 retention is at least 99.0% and KL at most 0.05;
9. N9 top-1 retention is at least 99.0%, every frozen group is at least 98.5%,
   and KL at most 0.02;
10. no extraction, legality, feature, full-forward, loss, finite-value, or
    serialization error occurs.

Failure ends N21 offline.  Passage permits only construction, parity audit, and
preregistration of one both-seat official-engine development panel against the
live control.  It does not authorize confirmation, submission, deployment,
rank, strength, or superhuman claims.  The live submission remains unchanged.
