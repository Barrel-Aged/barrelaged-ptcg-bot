# N20 Projection Execution Lock

Status: **frozen before grid evaluation**  
Protocol: `rocket-n20-trust-region-projection/v1`

## Immutable artifacts

- protocol SHA-256:
  `8f4628bf07f5a1abf21ba132bc9c7a7705031652a9d74560adbb509e1a9e952a`;
- projection evaluator SHA-256:
  `d612058c08f6c2dfbffd0f444300d95bd57e5d1d82c4718c881eab79ed0ac6f1`;
- live source model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- rejected N19 terminal model SHA-256:
  `b0f0e72f7eafbf2bd5e0e4eca1920398534ddef516b57750a3c1f58dbb7d467f`;
- N19 corpus audit SHA-256:
  `d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed`;
- N19 training report SHA-256:
  `ed88d68c63988e79b74e0c33966bcdfe219cabfb78d9b0fe19f1482df7ebbe43`;
- N19 result SHA-256:
  `cfb2560fca269d969a9490453db28b7f7a3bf6598e96864cfd886ecc1962f227`;
- N17 target-audit SHA-256:
  `c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

The evaluator additionally freezes and verifies every imported corpus,
feature, full-forward, and training support script by SHA-256.  It uses full
relational forward passes for N6/N9 retention and excludes all 170 N17 strict or
replicated target states, leaving the locked 469 unchanged N6 anchors.

The output directory
`artifacts/team_rocket_recovery/n20_trust_region_projection/projection_v1` did
not exist when this lock was written.

## Sole command and selection

```powershell
python scripts/evaluate_n20_trust_region_projection.py
```

The script accepts no arguments.  It evaluates diagnostic endpoints 0 and 1
plus the seven preregistered eligible alphas.  The only possible selected model
is the largest eligible alpha passing every teacher, fresh retention, N6, N9,
state, finite-value, and serialization gate.

No grid point may be added, rerun with modified support, or chosen by an
official-match outcome.  Passage grants only authority to design and lock one
both-seat official-engine development panel.  Deployment authority is always
false.
