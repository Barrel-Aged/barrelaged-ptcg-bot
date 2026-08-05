# N17 Development Build Lock

Status: **frozen before candidate bundle construction**  
Protocol: `rocket-n17-replicated-target-residual/v1`

- protocol SHA-256:
  `0a3348c41fd5fc821fbfda70280ee516d25c0425b45556a742afc56c9ef381b4`;
- offline result SHA-256:
  `0ffc6b8ecc6872059835552ff497433b663f8b63a9ee51abefec39389206413c`;
- training report SHA-256:
  `50aaac5e3531699489133790bdcbf0d566aea7acec23775ddd5b91d144fc02dd`;
- training provenance SHA-256:
  `1b14b12eef429f5019816d905dbd9c3e4c8d0e1da31265af8832b8423fca5789`;
- candidate model SHA-256:
  `381d6cb87a51fa1c8061f5e3ae441e446b0331bdd9c1f99e9118d4227e038772`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- residual runtime SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`;
- bundle/arm builder SHA-256:
  `b9a61933608ac453c2c8b3172a9169d6730784316d4ade8e992044543ad42912`.

The output directory was absent at lock time. The builder accepts no
arguments, preserves the exact Team Rocket deck and audited residual runtime,
and constructs exactly one `control`/`candidate` arm manifest. It has no match
or deployment authority.

```powershell
python scripts/build_n17_development_arms.py
```
