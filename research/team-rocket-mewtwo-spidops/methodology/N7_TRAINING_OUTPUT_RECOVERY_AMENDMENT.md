# N7 Training Output Recovery Amendment

Status: frozen after the training process exited and before any N7 runtime-parity or official-engine development game.

This amendment changes no data, split, model, seed, batch order, optimizer, hyperparameter, epoch definition, opponent, game seed, advancement threshold, router rule, or deployment condition in `N7_ARENA_MARNIE_SECOND_AWR_PROTOCOL.md`.

## Event

The preregistered trainer was launched once with the exact frozen command. The outer shell capture timed out after 3,604 seconds. Windows process `17480`, started at 2026-07-30 00:16:21 local time, remained alive and continued consuming CPU in the same process. It atomically wrote:

| Arm | Timestamp | Bytes | SHA-256 |
|---|---|---:|---|
| epoch 1 | 00:52:29 | 17,211,379 | `e47df4642c513fe776d36d96904cac6346873741b54265a7669707d1e3221dc9` |
| epoch 2 | 01:07:56 | 17,211,379 | `1ac0797a97fe1494ffec337d96b2305e4797205ffb52e89b29eff250bc8f07f2` |
| epoch 3 | 01:23:41 | 17,211,379 | `b906d64e41f56554d088dee37f21a71561c6badcb6941b88c1dc29273d99ff2f` |
| epoch 4 | 01:39:34 | 17,211,379 | `aa079ee3a76803ac6c3024861f42bf2f8c4fbc9f41ccb8a3d480ceb8a6591192` |

The same process then exited. It did not leave the final `model.pt` alias or `training_summary.json`; stdout epoch records were unavailable after the outer capture timed out. The most likely boundary is the detached epilogue/output path, but this document does not claim an unobserved exception cause.

## Recovery rule

The four epoch files are exactly the four development arms preregistered in the original protocol. The missing final alias has no scientific or deployment role: `epoch_04.pt` is already the final epoch byte artifact. Offline metrics were explicitly denied checkpoint-selection authority, so missing stdout metrics cannot alter which arms enter the panel.

The epoch files may advance to runtime parity only if `scripts/audit_n7_training_recovery.py` passes all of the following without modification:

1. source model, deck, trainer, protocol, and all four epoch hashes match the constants frozen in that audit;
2. exactly four epoch files exist and their timestamps are monotonic;
3. every checkpoint loads with the exact source architecture;
4. every tensor is finite;
5. every tensor outside the preregistered `mode=last` scope is byte-identical to the source model;
6. every epoch differs from the source and from its predecessor;
7. the recovery manifest is written before any official-engine panel game.

If any condition fails, N7 is rejected. The audit may not rewrite a checkpoint, synthesize `model.pt`, infer lost training metrics, or select an epoch.

## Unchanged evaluation authority

All four recovered arms still require the original runtime-parity gate and the locked 2,400-game official-engine panel. The live bot remains unchanged. No recovered checkpoint has deployment authority, and no strength claim follows from successful tensor recovery.
