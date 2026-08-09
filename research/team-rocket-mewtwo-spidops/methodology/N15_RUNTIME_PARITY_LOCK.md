# N15 Runtime Parity Lock

Status: **frozen before official-engine packaged-path parity execution**

- arm manifest SHA-256:
  `e46197852d7e88d069b54d9ecc5cc978b5d0d5b0c731cb34409a9d88c1fa34ef`;
- candidate bundle tree SHA-256:
  `982fa183a4ed963bda201c5dd01b55cf316e37848797441ef34d4848e0c7ff74`;
- verifier SHA-256:
  `21997b05d72da97819eabf30a0f4a2739bea2e9d4ff63fead377548511f6df95`.

The verifier runs four official-engine games per opponent per arm under seed
`815001`, compares every Rocket action from the strict development actor with
the packaged `NNAgent`, and requires at least 600 matched decisions with zero
mismatches. It records no outcome statistic and has no strength or deployment
authority.

Locked command:

```powershell
python scripts/verify_n15_runtime_parity.py
```
