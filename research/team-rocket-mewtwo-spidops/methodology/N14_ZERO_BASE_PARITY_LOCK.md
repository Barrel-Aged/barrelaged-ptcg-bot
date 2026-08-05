# N14 Zero-Residual Parity Lock

Status: **frozen before full support parity execution**

- Audit script SHA-256:
  `19e0ed7b5aa069f52573ac69b1b71f4113abbb1fd28cca0fbee98de369934657`.
- Zero-residual model SHA-256:
  `0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb`.
- Patched `net_v10.py` SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`.
- Live model/deck remain the N14 protocol hashes.

Locked command:

```powershell
python scripts/audit_n14_zero_residual_parity.py
```

All 639 N6R and 6,753 N9 support actions must match live labels and each other,
original state tensors must be exact, and finite relative-logit error must not
exceed `2e-5`. Failure forbids N14 training.

