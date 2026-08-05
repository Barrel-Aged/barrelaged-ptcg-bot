# N17 Training Execution Lock

Status: **frozen after exact zero-residual parity and before optimization**  
Protocol: `rocket-n17-replicated-target-residual/v1`

## Immutable implementation and evidence

- Protocol SHA-256:
  `0a3348c41fd5fc821fbfda70280ee516d25c0425b45556a742afc56c9ef381b4`.
- Target-audit SHA-256:
  `c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c`.
- Trainer SHA-256:
  `5501bb3e7e65a2382159160a71258f0274db4035500c8fa933e02a571fc5e6e9`.
- Trainer tests SHA-256:
  `25d98a31d49c788c62316ee6bce7c01fa71e011c855cd75eddf0cc3dab58846b`.
- Shared N12 support SHA-256:
  `843c12e0d2a1684d53ed27423c6b6c5cfe2dc38109c57b089b3d98e38cf181bd`.
- Shared live-extraction helper SHA-256:
  `4f233d40e625c33711ab87438388f8ba68f65f2e103c563bf621c790c9a657e7`.
- Shared N9 replay helper SHA-256:
  `62182583002847e7ed0e18011050f35319ad5661b73f3fb2d83c2564286ed1ad`.
- Zero-base model SHA-256:
  `de4679c136af57af9dc6818b3100c1e10d74ac5c32d615ae02019c897ec07a6c`.
- Residual-capable runtime SHA-256:
  `5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290`.
- Zero-parity report SHA-256:
  `6268b01da89e73ee3e1fbf21d9141100b86bd7a503a09eeeaf1f4ecdcea692f2`.
- Exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

The zero base has exactly 13,088 adapter parameters, an exact-zero final
projection, and every original live tensor unchanged. Independent runtime
comparison passed all 639 N6R and 6,753 N9 actions with zero label/top-action
mismatch and exactly zero finite relative-logit error. The three N17 tests pass,
including the immutable 48/122/469 partition, cell-balanced loss mass, rank-32
freeze boundary, and per-cell admission condition. The output directory was
absent when this lock was written.

## Locked command and terminal-only rule

```powershell
python scripts/train_n17_replicated_target_residual.py
```

No override is accepted. The run must execute exactly 4,000 steps with seed
`817002`, deterministic N6/N9 anchor orders derived as `817103` and `817204`,
four CPU threads, and the exact objective in the protocol and hashed trainer.
Only the step-4,000 terminal model may be saved. No early-stop, best-loss,
first-hit, alternate-rank, alternate-seed, or alternate-hyperparameter model is
authorized.

The trainer cannot read N15/N16 match outcomes and contains no match runner.
Every offline, serialization, asset, and full-runtime parity gate remains
mandatory. Training output has no strength or deployment authority. The fresh
official-engine development panel is authorized only if every offline gate
passes. The live bot remains unchanged.
