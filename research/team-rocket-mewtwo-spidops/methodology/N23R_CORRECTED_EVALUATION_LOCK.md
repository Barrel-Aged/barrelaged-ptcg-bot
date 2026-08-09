# N23R Corrected Evaluation Lock

Status: **frozen before the sole corrected held-out rerun**

- governing N23 protocol SHA-256:
  `b9eb93a9103a933b22e8f86c469af7ae39e06672e028643002c69ec295170171`;
- tie-correction amendment SHA-256:
  `99afca5165703ad1b60882486cba67021a776a160025a5a70305535f5c4d59c7`;
- train-only correction audit SHA-256:
  `4a33b1226829e760231c48477f227c8c78429b7d4d9cbc41e90377f4357238e4`;
- corrected evaluator SHA-256:
  `ea263cffbbf8320c5f193569caa1d36329031d78281871ee2d12e85ea77f6519`;
- quarantined original evaluation SHA-256:
  `873efb37a8652ceb03935af461883bb9abc50f1bb360a9199d295011559966fb`;
- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- N21 terminal SHA-256:
  `7e98ef2eb0933b1444183f8b9238344df3b17220af638a4b2045e5c6f5aeba2b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

The corrected output path
`artifacts/team_rocket_recovery/n23r_first_index_gate/EVALUATION_CORRECTED.json`
did not exist when this lock was written. The threshold remains `0.50`; the
only semantic correction is first-index tie resolution. The evaluator retains
every original N23 gate and verifies all source/support hashes.

## Sole command

```powershell
python scripts/evaluate_n23r_first_index_gate.py
```

This is the only corrected held-out rerun authorized for the family. Failure
closes it before runtime construction or official matches. Passage grants only
runtime build/parity and separately locked development-panel authority.
