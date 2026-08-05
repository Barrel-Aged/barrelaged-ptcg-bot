# N24 Training and Held-Out Execution Lock

Status: **frozen before the sole final N24 training run and before any N24 held-out metric**  
Protocol: `rocket-native-conservative-iql/v1`

## Frozen bytes

- protocol `N24_CONSERVATIVE_NATIVE_IQL_PROTOCOL.md` SHA-256:
  `76c9dd6634f033fe2bb7cd8c100d83340d5e6e06e6e68f8525df6b03d9b9a630`;
- trainer/evaluator `scripts/n24_native_iql.py` SHA-256:
  `1b372ea9e01f5c6a2d45befbaae8b1a2ab060dc746ec00fa12336488582d6be0`;
- tests `tests/test_n24_native_iql.py` SHA-256:
  `a7820fdd464224f1a52c430a76b04bfadf20c2fce29b792b36e58a1317b82d01`;
- train cache SHA-256:
  `e6e83143773462e52a0142e718a65c75ec93cc4abe0461f9437dcc42a332c3b0`;
- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- exact parity report SHA-256:
  `ebe70ea381a2f8eef38434ea8ca8ee2f08c867e03a497518520c17188758935e`.

All five N24 deterministic unit tests must pass before execution.

## Sole final training command

```powershell
python scripts/n24_native_iql.py train `
  --cache work/n24_native_iql/train.pt `
  --output artifacts/team_rocket_recovery/n24_native_iql/training_v1 `
  --threads 1
```

No exploratory checkpoint has evaluation or deployment authority. The sole
terminal checkpoint from this command is the only N24 held-out arm.

## Held-out cache and evaluation order

After final training completes and its checkpoint hash is frozen, generate the
two caches with the frozen script:

```powershell
python scripts/n24_native_iql.py cache --split validation `
  --output work/n24_native_iql/validation.pt --threads 1
python scripts/n24_native_iql.py cache --split test `
  --output work/n24_native_iql/test.pt --threads 1
```

Freeze both cache hashes before running either evaluation. Then:

```powershell
python scripts/n24_native_iql.py evaluate `
  --cache work/n24_native_iql/validation.pt `
  --heads artifacts/team_rocket_recovery/n24_native_iql/training_v1/iql_heads.pt `
  --output artifacts/team_rocket_recovery/n24_native_iql/validation.json `
  --threads 1
```

Do not execute the test command unless validation reports `admitted: true`.
If admitted, use the identical evaluator and terminal checkpoint against
`work/n24_native_iql/test.pt`, writing `test.json`.

No code, model, threshold, calibration, gate, or cache may be changed after a
held-out metric is opened. Any required correction rejects this execution and
must begin under a new protocol identifier. The live bot remains unchanged.
