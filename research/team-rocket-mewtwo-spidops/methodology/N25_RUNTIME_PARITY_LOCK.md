# N25 Runtime Parity Lock

Status: **frozen before packaged-path parity execution**

- arm manifest SHA-256:
  `defd9b9a02bbb221a8ef6d22b496d9dd37931870c8bb93001c32fdeaefe176a1`;
- candidate bundle tree SHA-256:
  `38ce5eaed5074544bf89fb723789ca39258dd9d734bf714f9803f5235e60dcbd`;
- verifier SHA-256:
  `2aa03002dd9ffb1ac15c7653f514c150358a48561fe33ab44858b44cb5a63355`.

Under seed `825000`, the verifier runs four official games per opponent per arm
and requires at least 600 packaged/strict matched Team Rocket decisions with
zero mismatches and no illegal actions. It records no outcome statistics and
has no strength or deployment authority.

```powershell
python scripts/verify_n25_runtime_parity.py
```
