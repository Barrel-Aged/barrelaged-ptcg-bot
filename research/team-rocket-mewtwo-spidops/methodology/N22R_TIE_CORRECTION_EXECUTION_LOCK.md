# N22R Tie-Correction Execution Lock

Status: **frozen before corrective train-only audit**  
Amendment: `rocket-n22-n23-first-index-tie-correction/v1`

- amendment SHA-256:
  `99afca5165703ad1b60882486cba67021a776a160025a5a70305535f5c4d59c7`;
- correction-audit script SHA-256:
  `94dd22a0b9d367e037dbb845ff012776b6b7667364b3f79c6b8b1afa336ce408`;
- original frozen N23 evaluator SHA-256:
  `cf8b082e63549c15508515ae690b28ea1f29062223fc8a1df950557bcdaec8ae`;
- quarantined N23 output SHA-256:
  `873efb37a8652ceb03935af461883bb9abc50f1bb360a9199d295011559966fb`;
- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- N21 terminal SHA-256:
  `7e98ef2eb0933b1444183f8b9238344df3b17220af638a4b2045e5c6f5aeba2b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

The output file
`artifacts/team_rocket_recovery/n22r_tie_corrected/TRAIN_ONLY_CORRECTION.json`
did not exist when this lock was written. The script loads N19 train records
only and applies the deployed first-index tie rule. It cannot open validation or
test records.

## Sole command

```powershell
python scripts/audit_n22r_first_index_ties.py
```

Passage authorizes only construction and freezing of the one corrected N23
held-out evaluator described by the amendment. No model, official match,
development panel, or deployment is authorized.
