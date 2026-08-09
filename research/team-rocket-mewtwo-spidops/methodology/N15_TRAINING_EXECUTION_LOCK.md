# N15 Training Execution Lock

Status: **frozen before N15 optimization or terminal model construction**  
Protocol: `rocket-n15-prospective-terminal-consolidation/v1`

## Immutable implementation

- protocol SHA-256:
  `f6da7eb8f3bf6d8f64e9d5b3670c0cadee4fe171f4b7525f1569f07b1af32beb`;
- N15 replay trainer SHA-256:
  `67c43e72b76621e8253421d7f836286d666353ada81cb016c7c1bd7ec52c1046`;
- N15 tests SHA-256:
  `e54abf59ffe4a395cfa3665ddd3a863e543a0466abcce6154ed784161eb5b922`;
- imported frozen N14 trainer SHA-256:
  `ecd003d7a716022d24ed17c9cf39d4074fd92becd5919c4d594f91af4c21dfb0`.

The exact deck, live model, zero base, residual runtime, zero-parity report,
N14 protocol/report/result, and four archived prefix checkpoints remain frozen
to the hashes listed in `N15_TERMINAL_CONSOLIDATION_PROTOCOL.md`.

Before this lock, Python compilation passed. Four focused N14/N15 tests and all
13 selected `test_n1*.py` tests passed. The N15 output directory did not exist.

## Locked command

```powershell
python scripts/train_n15_terminal_consolidation.py
```

No command-line override is accepted. The trainer must repeat the exact N14
2,000-step trajectory. Tensor-exact state equality and the archived correction
counts are mandatory at steps 62, 90, 149, and 286. Only step 2,000 may be
saved as the terminal candidate.

If a prefix differs, the run is invalid and no terminal candidate or match
panel is authorized. If the terminal model fails any unchanged offline gate,
N15 ends without official matches. Passing offline gates authorizes only the
locked development panel, never deployment or a strength claim.

If interrupted, preserve the partial directory without inspecting learned
outcomes. Any recovery must start from step 1 under a separately documented
lock and reproduce every archived prefix exactly. The live bot remains
unchanged.
