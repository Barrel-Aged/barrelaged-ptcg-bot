# N13 Calibrated Norm2 Result

Status: **complete negative offline result, 2026-08-01**. No N13 checkpoint
exists or is authorized for match testing, packaging, validation, or arena
submission.

## Locked result

N13 ran the sole preregistered 100x learning-rate calibration (`5e-5`) for all
400 steps over the same exact N12 support. Cache parity was exact on 7,392
decisions and the trainable scope remained four tensors/1,153 parameters.

The fit count progressed from 0/48 at step 1 to 11/48 at step 400:

- V110 turn 2: 1;
- V110 turn 4: 4;
- Recovery turn 2: 3;
- Recovery turn 4: 3.

The weakest authorized checkpoint required 12/48. Because fit12 was never
reached, the output contains only `training_report.json`, no model, and
`admitted: []`. No official-engine panel is permitted.

The immutable report is
`artifacts/team_rocket_recovery/n13_calibrated_norm2_adapter/training_v1/training_report.json`,
SHA-256
`d99d5552a069f8e18b34c85644832b89970250fe80d2bba8dcdcbd8709e0a3ea`.

## Why extending this path is not justified

N13 was one correction short of fit12, but its last-step sampled preservation
KL had already reached `0.1717` on N6R anchors and `0.1057` on N9 anchors,
compared with final admission ceilings of `0.05` and `0.02`. Those minibatch
values are diagnostics rather than the full fixed gate, but they are far enough
outside the ceilings to reject the simplistic conclusion that a few more steps
would yield a safe candidate.

The direct correction loss fell substantially (`1.9557` to `0.8099`) while
broad drift grew with it. This identifies a representation problem: a global
LayerNorm/action-head transformation can move the desired roots, but cannot
localize the intervention on the existing contextual channels under the fixed
trust region.

The next experiment must add a small explicitly residual, state/option-
conditioned adapter on frozen contextual option embeddings, initialized to
exactly zero and trained against both correction and broad preservation
supports. It may not extend N13, choose a different N13 step, or loosen safety
gates. The live bot remains unchanged.

