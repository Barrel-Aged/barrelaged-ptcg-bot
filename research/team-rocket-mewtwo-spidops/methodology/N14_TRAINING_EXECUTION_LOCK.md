# N14 Training Execution Lock

Status: **frozen after exact zero-residual parity and before optimization**  
Protocol: `rocket-n14-localized-contextual-residual/v1`

## Immutable implementation and evidence

- Protocol SHA-256:
  `2cbf0206af920072705421a349b7d13476766a1eff61dbd0494844bd0ed6821d`.
- Trainer SHA-256:
  `ecd003d7a716022d24ed17c9cf39d4074fd92becd5919c4d594f91af4c21dfb0`.
- Trainer tests SHA-256:
  `8f32296d7ba39e782542de536a658211d770fb60fc9bbb19f8857f064c51583c`.
- Zero-base model SHA-256:
  `0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb`.
- Patched runtime SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`.
- Zero-parity report SHA-256:
  `c8638bd6d5a879d32eb88bba30f7ddeb334d222e9b295c64f0ca2619096611d5`.
- Live model/deck, transfer audit, N9 parity/tree, and N13 report remain the
  hashes frozen in the protocol.

The zero base has exactly 6,928 adapter parameters, zero final projection, and
preserves every original tensor. Direct independent runtime comparison passed
all 639 N6R and 6,753 N9 actions with zero label/top mismatch and exactly zero
finite relative-logit error. Thirteen selected builder/residual/cache/estimator
tests pass.

## Locked command

```powershell
python scripts/train_n14_localized_contextual_residual.py
```

No override is accepted. The run must execute exactly 2,000 steps with the
single rank-16 architecture and fixed objective. Only first-hit
fit12/fit24/fit36/fit48 checkpoints may be created under
`artifacts/team_rocket_recovery/n14_contextual_residual/training_v1`.

Every offline and full-runtime parity gate remains mandatory. Training output
has no strength or deployment authority. If interrupted, preserve the partial
directory unopened and use a separately frozen from-step-1 recovery with exact
threshold-prefix reproduction. The live bot remains unchanged.

