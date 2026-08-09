# N17 Zero-Residual Parity Lock

Status: **frozen before full support parity execution**  
Protocol: `rocket-n17-replicated-target-residual/v1`

- Audit script SHA-256:
  `45d165c54f42cf88bd677103d6a2c9d8cb0a61222e9a89987173ffbe875e9cca`.
- Zero-residual model SHA-256:
  `de4679c136af57af9dc6818b3100c1e10d74ac5c32d615ae02019c897ec07a6c`.
- Residual-capable `net_v10.py` SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`.
- Protocol SHA-256:
  `0a3348c41fd5fc821fbfda70280ee516d25c0425b45556a742afc56c9ef381b4`.
- Target-audit SHA-256:
  `c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c`.
- Live model and exact Team Rocket deck remain the frozen protocol hashes.
- The parity-output path was absent when this lock was written.

Locked command:

```powershell
python scripts/audit_n17_zero_residual_parity.py
```

All 639 N6R and 6,753 N9 support actions must match live labels and each other;
all original state tensors must be exact; the adapter must have exactly 13,088
parameters and an exact-zero final projection; and finite relative-logit error
must not exceed `2e-5`. Failure forbids N17 training.
