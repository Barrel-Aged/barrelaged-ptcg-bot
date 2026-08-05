# N23 Evaluation Execution Lock

Status: **frozen before held-out evaluation**  
Protocol: `rocket-n23-heldout-high-margin-gate/v1`

## Immutable artifacts

- protocol SHA-256:
  `b9eb93a9103a933b22e8f86c469af7ae39e06672e028643002c69ec295170171`;
- evaluator SHA-256:
  `cf8b082e63549c15508515ae690b28ea1f29062223fc8a1df950557bcdaec8ae`;
- live model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`;
- rejected N21 terminal SHA-256:
  `7e98ef2eb0933b1444183f8b9238344df3b17220af638a4b2045e5c6f5aeba2b`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- N19 corpus audit SHA-256:
  `d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed`;
- N21 result SHA-256:
  `53683ec3d968e59d4343c5b7294aec00d46b2e8d0a49ed7a6d54d6f4890b1e98`;
- N22 audit SHA-256:
  `1dd3904da69914c322fe6d4b0f4337d1a4ed6146533519353431c9f70c999dd8`;
- N22 result SHA-256:
  `df2bbb6343999ad2db77ea3e49256a9cac10cad838d768e353b4787ac6833b5d`;
- N17 target-audit SHA-256:
  `c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c`.

The evaluator verifies all imported extraction, public-BC, attack-guard,
full-forward, live-AWR, N9, and N6 support files by SHA-256. The only gate rule
is top disagreement plus N21 deployed margin `>=0.50`; it contains no threshold
grid or fit procedure.

The output file
`artifacts/team_rocket_recovery/n23_high_margin_gate/EVALUATION.json` did not
exist when this lock was written. N19 held-out examples had not been
reconstructed by N22 and were not opened before this execution lock.

## Sole command

```powershell
python scripts/evaluate_n23_high_margin_gate.py
```

The script accepts no arguments. It performs no training or model mutation.
Failure ends this gate family before runtime construction or official matches.
Passage grants only runtime build/parity and separately locked development-panel
authority; deployment authority remains false.
