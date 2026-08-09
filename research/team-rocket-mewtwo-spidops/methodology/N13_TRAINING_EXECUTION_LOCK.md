# N13 Training Execution Lock

Status: **frozen before N13 optimization**  
Protocol: `rocket-n13-calibrated-norm2-adapter/v1`  
Frozen: 2026-08-01

## Immutable implementation

- Protocol SHA-256:
  `212a276849b4bcb2be6d9dd69c69effd94096c113001fb03d3c33fe3b800170d`.
- N13 single-path runner SHA-256:
  `a5a27047dbf58dab3bffd8b7e8134d996b6a0946e4977a6a453792c931187182`.
- N13 runner tests SHA-256:
  `dd93a8ac2afa4c06c8d1f2ecf2f788fc06dbab3035de75e3325b4083a605d843`.
- Frozen cache/objective implementation SHA-256:
  `843c12e0d2a1684d53ed27423c6b6c5cfe2dc38109c57b089b3d98e38cf181bd`.
- Live/control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- N12 transfer-audit SHA-256:
  `9ef9429934d7d94621406181f61c2a97d57b3409296235373f027efc9e8ed313`.
- N12 v3 result report SHA-256:
  `4a32577c7dbfaec481bf3993de3d89985f0043a7c3483baca6249fb52a5b4947`.
- N9 parity SHA-256:
  `ebe70ea381a2f8eef38434ea8ca8ee2f08c867e03a497518520c17188758935e`.
- N9 replay-tree SHA-256:
  `f430676aa3c7786e969d35f0725c70015742adcef5d4e4d4408293e0655cd7ae`.

Eleven selected calibration/cache/estimator tests pass. The unchanged cache
implementation previously reproduced all 7,392 relative-logit vectors exactly;
the N13 command must independently repeat that gate before its optimizer is
constructed. The opaque invalid v2 directory is never read.

## Locked command

```powershell
python scripts/train_n13_calibrated_norm2_adapter.py
```

No command-line argument is accepted. Exactly one learning rate (`5e-5`) and
one 400-step path are authorized. Output is exclusively
`artifacts/team_rocket_recovery/n13_calibrated_norm2_adapter/training_v1`.
Only first-hit fit12/fit24/fit36/fit48 models may be written; all fixed offline
gates are mandatory. Training output has no match-strength or deployment
authority.

If interrupted, preserve the partial directory unopened, freeze a recovery
lock, restart from step 1, and require exact reproduction of any complete
threshold prefix. The current live bot remains unchanged.

