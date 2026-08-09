# N22 Audit Execution Lock

Status: **frozen before analysis**  
Protocol: `rocket-n22-train-only-disagreement-audit/v1`

## Immutable artifacts

- protocol SHA-256:
  `337d2a37dda170e0c856532a398bf7d26ad7be4e7d94a3adee31ece11edf22d3`;
- audit script SHA-256:
  `e5377d671fabc1141e28af05cea4b8ef7f418fad843822c5b0f61d988c1d8c56`;
- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- rejected N21 terminal SHA-256:
  `7e98ef2eb0933b1444183f8b9238344df3b17220af638a4b2045e5c6f5aeba2b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- N19 corpus audit SHA-256:
  `d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed`;
- N21 training report SHA-256:
  `8f51c4ad18558ef7af48ca7a08fd9eb2e059a7e5f94290b0eb0a1f345357a0f6`;
- N21 result SHA-256:
  `53683ec3d968e59d4343c5b7294aec00d46b2e8d0a49ed7a6d54d6f4890b1e98`.

The script additionally verifies every imported N19 extraction, public-BC,
N12 full-forward, live-AWR, N9, and N6 support file by SHA-256. Its source code
contains no path for N19 validation or test reconstruction; only records whose
frozen manifest split is `train` are loaded. Historical N9 is explicitly
diagnostic preservation evidence, not a held-out gate.

The output file
`artifacts/team_rocket_recovery/n22_disagreement_gate/TRAIN_ONLY_AUDIT.json`
did not exist when this lock was written.

## Sole command

```powershell
python scripts/audit_n22_disagreement_gate_feasibility.py
```

The script accepts no arguments and performs no training or model mutation.
Only the fixed summaries and feasibility gates in the protocol may be used.
Regardless of result, development and deployment authority remain false.
