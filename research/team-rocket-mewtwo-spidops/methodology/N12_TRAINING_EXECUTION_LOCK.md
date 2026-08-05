# N12 Training Execution Lock

Status: **frozen after full-data preflight and before optimization**  
Protocol: `rocket-n12-counterfactual-margin-adapter/v1`  
Frozen: 2026-08-01

## Immutable substrate and implementation

- Protocol SHA-256:
  `bb9ecd5d77092ec391abceb46df16d883f1bf08d0fc1280cf6b967629b596396`.
- Trainer SHA-256:
  `ebf7d2bdf19b63a7cbe1865b4ac0b5cee759d769dc53f427ee3c973ed3ce5f7b`.
- Trainer tests SHA-256:
  `686e18cf5bc0cd6f23c5478976ffa64625d31ee4d071b092112b836b0d221771`.
- Frozen live/control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Frozen Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.
- Transfer-audit SHA-256:
  `9ef9429934d7d94621406181f61c2a97d57b3409296235373f027efc9e8ed313`.
- N9 exact-agent parity SHA-256:
  `ebe70ea381a2f8eef38434ea8ca8ee2f08c867e03a497518520c17188758935e`.
- N9 combined replay-tree SHA-256:
  `f430676aa3c7786e969d35f0725c70015742adcef5d4e4d4408293e0655cd7ae`.

## Pre-optimization evidence

The full read-only preflight completed before this lock:

- 639 exact N6R feature roots loaded, including exactly 48 hard-admissible
  corrections and 591 preservation roots;
- 6,753 exact N9 preservation actions loaded from the deterministic test split,
  comprising 3,456 V110 and 3,297 Recovery examples from 119 decisive games;
- frozen live action reproduced on 639/639 N6R roots;
- frozen live action reproduced on 6,753/6,753 N9 examples;
- zero feature, legality, actor-information, or serialization errors.

The following tests passed before freezing:

- four-cell correction weighting has exactly equal cell mass;
- smooth margin loss decreases monotonically with achieved margin;
- deterministic cyclic sampling is repeatable and complete per cycle;
- empty or over-broad state changes fail the permitted-key check;
- the inherited clustered counterfactual estimator tests remain green.

## Locked command

Run exactly:

```powershell
python scripts/train_n12_counterfactual_margin_adapter.py
```

The trainer must refuse any changed locked hyperparameter or existing output
directory. It runs exactly 400 steps and saves only the first checkpoints to
reach the preregistered 12/24/36/48 correction thresholds. Offline gates may
exclude checkpoints but may not select a different step.

## Interruption rule

The trainer has no mid-run resume authority. If interrupted before the atomic
final report exists, archive the incomplete output directory without opening
checkpoint metrics, rerun from step 1 with the exact locked command, and require
the rerun's threshold-step sequence and model hashes to reproduce any complete
archived prefix before using it. The live bot remains unchanged throughout.

Training output authorizes only construction of an N12 development-arm
manifest for checkpoints whose fixed offline gates pass. It has no deployment
or strength-claim authority.

