# N10 Power Interruption and Resume Lock

Status: **safely stopped; frozen before any N10 outcome inspection**  
Interruption: 2026-07-30, at the user's request before host power-off

## Preserved state

All N10 collector and opponent-worker processes were identified by exact
command line and terminated. No N10 arm win count, replay outcome, or partial
comparison was opened before this lock.

Two shards had completed and remain immutable inputs:

| Shard | Rows | JSONL SHA-256 | Manifest SHA-256 |
|---|---:|---|---|
| `v110_01` | 800 | `87511aa68bf6ccdc928307fc016c6ef23d1e051c2af048a231261ccf02154ae6` | `ddd4c0ae884ac24b585fa99adbc24028aca1d898766abce111b96a9013d96870` |
| `v110_02` | 800 | `2bea5ca36d4062e2ec91c1b523828c062952fb4d15aa7841683ac248ef03b666` | `c2292db87fd2214f5b46dfe93b07f7bb9e1c15633d2c4d625b3e1db913b955c1` |

The two incomplete Recovery files were closed, hashed, and moved out of the
panel input namespace to
`artifacts/team_rocket_recovery/n10_epoch4_confirmation/panel/interrupted_20260730_poweroff/`:

| Archived prefix | Rows | SHA-256 |
|---|---:|---|
| `recovery_01.partial.jsonl` | 551 | `b34a83b9361dabcdde0666d8e5c8c1e6a32ea92aa1c89113b466646bccd305c0` |
| `recovery_02.partial.jsonl` | 558 | `2628f1a59cd45470a02942b2c9d2605b3545ed36fc3a9169c0604d8c6dd3a9bc` |

The original incomplete files have no sidecar manifests and are forbidden
from analysis.

## Locked resumption procedure

On the next powered session:

1. Verify the four completed/archived hashes above, the frozen arm manifest
   SHA `d60ab7cbd53a49cc9a924f8a9b8d7445937f7db4b9c80d90ef7a04c007bdf443`,
   and original collector SHA
   `99882f0cdc4e722a03f45b1bf54bf2cf71e963ab0c1395b5e41b429b91f9a589`.
2. Re-run only the two Recovery shards from game 1 with the original collector,
   original seeds 799201 and 799202, original 400 games per arm, and original
   output names `recovery_01.jsonl` and `recovery_02.jsonl`. Do not replace or
   extend seeds and do not inspect outcomes.
3. Run `scripts/verify_n10_resumed_prefixes.py`, SHA-256
   `a4d35229a528708558d7a0546c15fdf4ac24a22ed1417ac3d9af817d6508b170`.
   It must prove that all 551 and 558 archived rows are byte-identical to the
   corresponding prefixes of the restarted 800-row shards.
4. If either prefix differs, N10 fails operationally and its analyzer must not
   run. A new full experiment would require new preregistered seeds.
5. Only if both prefixes match exactly and all four manifests are valid may the
   frozen N10 analyzer run on the four full shards.

This recovery rule cannot favor an arm: the power-off decision was external,
no outcomes were inspected, the same seeds and schedules are retained, and
the regenerated prefixes must reproduce byte for byte. The live arena bot and
its exact Team Rocket deck were never modified.

