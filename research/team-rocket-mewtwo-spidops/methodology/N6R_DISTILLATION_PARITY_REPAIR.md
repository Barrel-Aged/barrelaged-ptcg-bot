# N6R distillation parity repair (A1)

Status: frozen execution amendment, 2026-07-28.

This amendment changes only the lossless encoding boundary between the valid
N6R exact-feature rollout dataset and the frozen neural blueprint. It does not
change the deck, base model, rollout data, teacher estimator, train/selection
split, four arm definitions, optimization hyperparameters, development panel,
or advancement gates in `N6R_EXACT_FEATURE_BELIEF_ROLLOUT_PROTOCOL.md`.

## Fail-closed discovery

The first distillation invocation stopped before creating any model or output
directory. Its training encoder reported 399 rows, 392 encoded examples, and
7 baseline-reproduction mismatches. All four arm processes failed at the same
guard, so no partial student is eligible for evaluation.

The mismatch was caused by two execution differences, diagnosed solely from
baseline parity rather than teacher outcomes:

1. The old trainer changed the saved `opt_avail` feature before inference when
   ATTACK and END were both legal. The deployed `NNAgent` instead evaluates the
   original feature and masks END in the resulting logits. The pre-forward
   mutation moved unrelated logits and caused six mismatches.
2. Collection used one Torch inference thread. The attempted trainer used two;
   one near-tied root moved by about `1.9e-6` and changed argmax. Exact-feature
   distillation now requires one thread, matching collection.

Using the unmodified saved feature, the deployment-equivalent post-logit guard,
and one inference thread reproduces all 399/399 training baselines and all
240/240 independent-selection baselines exactly.

## Frozen repaired execution

- trainer: `scripts/train_belief_rollout_distillation_n6.py`;
- trainer SHA-256:
  `3c559b00c4d59c018295406e84dc78f2e19b5014b9174cb23c8db63790020a7f`;
- amended immutable-arm builder:
  `scripts/build_belief_rollout_n6r_arms.py`;
- arm-builder SHA-256:
  `2c92c1b05db791a1b3a13707e4fdb2548f5760ad9fcba4d57212a88ce386296a`;
- required argument: `--threads 1` for N6R v2 inputs;
- runtime guard: mask END logits only after network evaluation when an ATTACK
  option is legal;
- focused regression result: 7/7 standard-library unit tests passed;
- lossless parity audit: 399/399 training and 240/240 selection roots passed.

The historical trainer hash
`470f95e0bb9fabe3d6c794cd607a3a4f2878f39d3ca0ff6e57ddd6d348e988a3`
remains the record of the rejected invocation and is not eligible to produce an
N6R student.

## Leakage and reuse decision

The independent selection summary had been opened before the trainer failed:
17/240 admissible changes with 42.08% split-half agreement. The repair was
defined entirely by exact baseline reproduction and the already-frozen live
agent operation; it did not use action values, teacher-best identities, arm
metrics, or match outcomes. Therefore the sealed train and selection files are
reused without relabeling or recollection. The original arm grid and all later
strength gates remain unchanged. This is an amended experiment and must be
reported as such.
