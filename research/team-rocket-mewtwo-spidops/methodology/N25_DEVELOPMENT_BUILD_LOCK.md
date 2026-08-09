# N25 Development Build Lock

Status: **frozen before candidate bundle construction**

- protocol SHA-256:
  `82ba2a57736e396ecda5c2fdcda1a9577d68be25de88f4c2ddb4f098d86e78a0`;
- source-base correction SHA-256:
  `aadd14c675e44a641b9cb0022c111aa578e516206b4f1b556fc918ff5a3d9e33`;
- offline build report SHA-256:
  `b94d281479b4be1cb29b7352f27c3b91ec36f7503edd3f5b3a5ede556b7a81ce`;
- candidate model SHA-256:
  `e1d6567dfefee9c7b408b9438153e75e75f613864d7c0a2da021f7f5c63ebf04`;
- bundle/arm builder SHA-256:
  `d4e9bf85bb2cb4e8240c8c62661b9ae150d13b91f4ecf5188c6acaa004d67cb8`.

The builder accepts no arguments, preserves the exact qualifying Team Rocket
deck and audited residual-capable runtime, and constructs exactly one
`control`/`candidate` arm manifest. It has no match or deployment authority.

```powershell
python scripts/build_n25_development_arms.py
```
