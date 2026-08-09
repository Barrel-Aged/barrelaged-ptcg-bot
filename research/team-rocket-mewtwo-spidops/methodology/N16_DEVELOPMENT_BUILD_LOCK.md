# N16 Development Build Lock

Status: **frozen before candidate bundle construction**

- protocol SHA-256:
  `f7b74e00ba0f7250550bca21d4454bcf903169343e82d0dee60cab326d0ba190`;
- projection report SHA-256:
  `cb8b8f970ab913362180f89a396d19adca7a300bf65be3ed64f4990d08ee525a`;
- candidate model SHA-256:
  `b4f35c7816097800b0c7265691f5f650bc9b467a4258612930d0fef07afcbcf7`;
- bundle/arm builder SHA-256:
  `8709c6d587797d260efad9535f6265dd2d5f4640feda3e3da3fdc8cec95a4468`.

The builder accepts no arguments, preserves the exact Team Rocket deck and
audited residual runtime, and constructs exactly one `control`/`candidate` arm
manifest. It has no match or deployment authority.

```powershell
python scripts/build_n16_development_arms.py
```
