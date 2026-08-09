# N19 Training Execution Lock

Status: **frozen before training**  
Protocol: `rocket-n19-retention-constrained-leader-bc/v1`

## Immutable artifacts

- protocol SHA-256:
  `006fd34be9d3770ed8025c2db2a5c0f21c6aa5f2e4dbd4baa9cde50ad37e153e`;
- corpus audit SHA-256:
  `d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed`;
- corpus auditor SHA-256:
  `86f6e12bb96c16ebfa0980e27974bff881537cb8e750e9c5adc6a688bfda58c1`;
- training script SHA-256:
  `3928abaf17457d05e02c7d9734f69f819705601051154bc78236a981868de7b7`;
- source live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

The corpus contains the exact preregistered 9,483 teacher and 6,240 retention
training targets, plus disjoint validation and test splits.  The output
directory
`artifacts/team_rocket_recovery/n19_retention_teacher/training_v1` did not
exist when this lock was written.

## Sole command

```powershell
python scripts/train_n19_retention_constrained_bc.py
```

The script accepts no arguments.  It runs exactly four epochs and 396 optimizer
steps under seed `819001`.  Only the terminal epoch-4 model is eligible.  An
interruption leaves no authorized substitute checkpoint; any recovery must be
durably specified before a rerun.

The training report may adjudicate the fresh teacher/retention corpus gates.
Even if every such gate passes, the N6/N9 exact-Rocket anchor audit remains
mandatory and development-panel authority remains false until that audit is
complete.  No training outcome grants deployment or submission authority.
