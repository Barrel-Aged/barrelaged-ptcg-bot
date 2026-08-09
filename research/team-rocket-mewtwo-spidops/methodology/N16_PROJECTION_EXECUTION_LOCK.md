# N16 Projection Execution Lock

Status: **frozen before analytic scale calculation or model construction**

- protocol SHA-256:
  `f7b74e00ba0f7250550bca21d4454bcf903169343e82d0dee60cab326d0ba190`;
- projection builder SHA-256:
  `531ffa91a9bcbe0e9b01693ee13bdbccae121f661a1e91b5868a7d2937d08f9a`;
- focused tests SHA-256:
  `7779326d979242b7233a89f68b3f7711d8010f3e3adeb57e0b23426310ab4c4d`;
- source terminal model SHA-256:
  `354a41f5d4146ae2b48b3935653a0e47f535150eebcd37e90aa866d98d0c5ee7`;
- N15 training report SHA-256:
  `b508381af55d3f86654ab527622bec333696fa7ab37b05fb06f5b71933d92ce1`.

The builder contains no N15 development-panel path, row, analysis, seed, or
outcome input. It accepts no arguments. Seven focused N14-N16 tests and all 16
selected `test_n1*.py` tests pass. The output directory did not exist at lock
time.

Locked command:

```powershell
python scripts/build_n16_minimum_scale_projection.py
```

Exactly one scale may be produced by the frozen inequality intersection and
decimal upward-rounding rule. No alternative scale may be inspected or built.
Failure of feasibility or any offline gate ends N16 before official match play.
The live bot remains unchanged.
