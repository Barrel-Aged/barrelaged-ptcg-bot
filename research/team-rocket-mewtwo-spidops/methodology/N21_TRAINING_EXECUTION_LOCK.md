# N21 Training Execution Lock

Status: **frozen before training**  
Protocol: `rocket-n21-top-action-margin-bc/v1`

## Immutable artifacts

- protocol SHA-256:
  `2a18a5f8ad3242e5d4c13ff132187121563b20b27f384597b91dc25d607cf839`;
- training script SHA-256:
  `22fdd063349f4a57732eb8607359db0707962dea2a466d2a5212ec67488de85e`;
- live source model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- N19 corpus audit SHA-256:
  `d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed`;
- rejected N19 result SHA-256:
  `cfb2560fca269d969a9490453db28b7f7a3bf6598e96864cfd886ecc1962f227`;
- rejected N20 report SHA-256:
  `83f8094ecfacbeb6c73caaf899af2fb7cbbfc9779ce5f4ec9d77bb23218319e0`;
- rejected N20 result SHA-256:
  `84fa85599e38cb50dc03265e8d090666467993ee34681dda62b46f82861c1d8c`;
- N17 target-audit SHA-256:
  `c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c`.

The trainer freezes and verifies by SHA-256 its N19 trainer, N20 evaluator,
N19 corpus auditor, public-BC support, N12 full-forward support, live-AWR
support, N9 support, and N6 support. It reconstructs the locked N19 splits,
uses full relational forward passes for N6/N9, and excludes all 170 N17 target
states, leaving exactly 469 unchanged N6 anchors.

The output directory
`artifacts/team_rocket_recovery/n21_top_action_margin/training_v1` did not
exist when this lock was written.

## Sole command

```powershell
python scripts/train_n21_top_action_margin_bc.py
```

The script accepts no arguments. It runs exactly two epochs and 198 optimizer
steps under seed `821001`, starting from the exact live model. Only the terminal
epoch-2 model is eligible. An interruption leaves no authorized substitute
checkpoint; recovery or rerun requires a durable amendment written before any
new execution.

No training outcome grants deployment or submission authority. Passage of
every offline gate grants only authority to construct, parity-audit, and
preregister one both-seat official-engine development panel against the live
control.
