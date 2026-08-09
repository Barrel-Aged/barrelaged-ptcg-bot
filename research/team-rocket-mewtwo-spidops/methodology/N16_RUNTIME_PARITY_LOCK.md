# N16 Runtime Parity Lock

Status: **frozen before packaged-path parity execution**

- arm manifest SHA-256:
  `3cb430f6e281004b1671d0df3941e6a14386278d789c2e153b51221fe6e4bf72`;
- candidate bundle tree SHA-256:
  `d4732ea468b9046abffd808937935963cba4091f180ae72704d03e66a6bfcf02`;
- verifier SHA-256:
  `fb15f08e6e93b5f4da6d121a04c2b8c34ae9ce58fb128e8cf9f10712e3f3eadc`.

Under seed `816000`, the verifier runs four official games per opponent per arm
and requires at least 600 packaged/strict matched Rocket decisions with zero
mismatches. It records no outcome statistics and has no strength authority.

```powershell
python scripts/verify_n16_runtime_parity.py
```
