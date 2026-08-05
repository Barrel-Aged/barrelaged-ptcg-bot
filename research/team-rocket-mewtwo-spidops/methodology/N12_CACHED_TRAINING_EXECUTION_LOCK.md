# N12 Cached Training Execution Lock

Status: **frozen after cached-path parity and before cached optimization**  
Protocol: `rocket-n12-counterfactual-margin-adapter/v1`  
Execution: `rocket-n12-cached-final-block-execution/v2`  
Frozen: 2026-08-01

## Immutable evidence and code

- Scientific protocol SHA-256:
  `bb9ecd5d77092ec391abceb46df16d883f1bf08d0fc1280cf6b967629b596396`.
- Cached execution amendment SHA-256:
  `8085d894e6527fbe6b78703c17aadd284705f2640fc830fbe4e9dec56e147459`.
- Cached trainer SHA-256:
  `06d90c3805a18bba004cf7c23d6e0892d1c9b0d5a5fca587afbe0fd94548dc3b`.
- Cached trainer tests SHA-256:
  `b0d766b466a66181b259bdee2521b6810763dcca869b8e15937c4b5efecee773`.
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

## Cached-path admission evidence

Before this lock, a fresh read-only run rebuilt every example and cache:

- N6R: 639 examples, 0 reference-action mismatches, 0 cached top-action
  mismatches, maximum finite relative-logit error
  `0.00001049041748046875`;
- N9: 6,753 examples, 0 reference-action mismatches, 0 cached top-action
  mismatches, maximum finite relative-logit error
  `0.0000171661376953125`;
- both maxima pass the frozen `0.00002` ceiling;
- deterministic logical-batch versus microbatch gradients passed with absolute
  tolerance `2e-6` and relative tolerance `2e-5`;
- all nine selected N12/estimator unit tests passed.

The invalid v1 output is an empty preserved directory and contains no observed
training metric or checkpoint. It has no authority.

## Locked command

Run exactly:

```powershell
python scripts/train_n12_counterfactual_margin_adapter.py
```

The command must create only
`artifacts/team_rocket_recovery/n12_counterfactual_margin_adapter/training_v2_cached`.
It must run 400 steps even if every threshold is reached early. Only first-hit
fit12/fit24/fit36/fit48 checkpoints may exist. The final report and offline
gates are descriptive/safety evidence, not match strength.

## Interruption rule

If interrupted, preserve the partial directory without inspecting its metrics,
rerun from step 1 under a separately frozen recovery lock, and require exact
threshold-step and checkpoint-hash reproduction for every complete prefix.
The current live bot remains byte-unchanged.

