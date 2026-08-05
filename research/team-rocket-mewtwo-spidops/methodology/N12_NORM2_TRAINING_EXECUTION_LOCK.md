# N12 Norm2 Training Execution Lock

Status: **frozen after exact v3 cache parity and before v3 optimization**  
Protocol: `rocket-n12-norm2-margin-adapter/v3`  
Frozen: 2026-08-01

## Immutable inputs and implementation

- V3 protocol SHA-256:
  `d4a4b66ef45190feff09b70ca5eac271dcbc9a16ba492311379fae98469ad51b`.
- V3 trainer SHA-256:
  `843c12e0d2a1684d53ed27423c6b6c5cfe2dc38109c57b089b3d98e38cf181bd`.
- V3 tests SHA-256:
  `e660a263c7a3d0fce4debfba7c48d1ec62c8410b2b88ca27f9496650510e2a0c`.
- Live/control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Transfer-audit SHA-256:
  `9ef9429934d7d94621406181f61c2a97d57b3409296235373f027efc9e8ed313`.
- N9 parity SHA-256:
  `ebe70ea381a2f8eef38434ea8ca8ee2f08c867e03a497518520c17188758935e`.
- N9 replay-tree SHA-256:
  `f430676aa3c7786e969d35f0725c70015742adcef5d4e4d4408293e0655cd7ae`.

## Pre-optimization gates

- 639/639 N6R reference actions reproduce; cached path has zero top-action
  mismatches and exactly zero finite relative-logit error.
- 6,753/6,753 N9 reference actions reproduce; cached path has zero top-action
  mismatches and exactly zero finite relative-logit error.
- Trainable scope is exactly 1,153 parameters in exactly four tensors:
  `encoder.layers.2.norm2.weight`, `encoder.layers.2.norm2.bias`,
  `logit_head.weight`, and `logit_head.bias`.
- Logical-batch/microbatch gradient equivalence and all selected estimator tests
  pass (nine tests total).
- V1 and v2 invalid outputs are excluded; the v2 partial remains opaque and
  uninspected.

## Locked command

```powershell
python scripts/train_n12_counterfactual_margin_adapter.py
```

The only authorized output is
`artifacts/team_rocket_recovery/n12_counterfactual_margin_adapter/training_v3_norm2`.
The run must complete all 400 steps and may save only first-hit
fit12/fit24/fit36/fit48 checkpoints. Offline admission has no strength or
deployment authority.

If interrupted, preserve the partial output unopened, freeze a recovery lock,
rerun from step 1, and require exact reproduction of every complete threshold
prefix before continuing. The live bot remains unchanged.

