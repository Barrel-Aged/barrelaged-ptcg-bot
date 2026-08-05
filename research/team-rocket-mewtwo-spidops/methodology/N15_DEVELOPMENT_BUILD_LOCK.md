# N15 Development Build Lock

Status: **frozen before candidate-bundle or arm-manifest construction**

- N15 protocol SHA-256:
  `f6da7eb8f3bf6d8f64e9d5b3670c0cadee4fe171f4b7525f1569f07b1af32beb`;
- N15 training report SHA-256:
  `b508381af55d3f86654ab527622bec333696fa7ab37b05fb06f5b71933d92ce1`;
- admitted terminal model SHA-256:
  `354a41f5d4146ae2b48b3935653a0e47f535150eebcd37e90aa866d98d0c5ee7`;
- arm builder SHA-256:
  `4e9b59f4916b6430dd1a70e06b05852840f536a3a185baaef0516b558d0ed1ca`.

The builder accepts no overrides and may create exactly one candidate bundle
plus a two-arm `control`/`candidate` manifest. It copies the already-audited
residual-capable runtime, replaces only `model.pt` with the admitted terminal
model, preserves the exact Team Rocket deck, and records provenance. The build
has no match, strength, or deployment authority.

Locked command:

```powershell
python scripts/build_n15_development_arms.py
```
