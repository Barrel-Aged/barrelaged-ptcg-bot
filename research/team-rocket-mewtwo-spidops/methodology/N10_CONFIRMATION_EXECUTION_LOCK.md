# N10 Confirmation Execution Lock

Status: **frozen before collection**  
Protocol: `rocket-n10-independent-epoch4-confirmation/v1`  
Frozen: 2026-07-30

## Immutable implementation

- Protocol document SHA-256:
  `c216f03514c93261eecb355c9fb878d133f76985c62c4e35d5eac5ac2cfd317f`.
- Arm builder SHA-256:
  `1ade359b8b8193925509e7d8ee2a4ade0f910d8781e51738d09f58cbb7c8d001`.
- Collector SHA-256:
  `99882f0cdc4e722a03f45b1bf54bf2cf71e963ab0c1395b5e41b429b91f9a589`.
- Analyzer SHA-256:
  `db11727af97580a7c67a06118fc7cbc3e223faa95bf7b1718701f16523fc94da`.
- Arm manifest SHA-256:
  `d60ab7cbd53a49cc9a924f8a9b8d7445937f7db4b9c80d90ef7a04c007bdf443`.
- Control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Candidate model SHA-256:
  `bdafc1ce36f7e572554930a1a9b1beff541a351ece09f934d0416b4ce269a020`.
- Exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

Both Python entry points compile and expose their argument parsers. The arm
builder verified the source N9 analysis, source arm manifest, model files, and
the frozen zero-mismatch packaged-runtime parity report before emitting the
two-arm manifest.

## Frozen execution

Each shard uses 400 games per arm (800 rows):

| Shard | Opponent | Seed | Output |
|---|---|---:|---|
| `v110_01` | `AshMarnieGeneralV110_extracted` | 799101 | `artifacts/team_rocket_recovery/n10_epoch4_confirmation/panel/v110_01.jsonl` |
| `v110_02` | `AshMarnieGeneralV110_extracted` | 799102 | `artifacts/team_rocket_recovery/n10_epoch4_confirmation/panel/v110_02.jsonl` |
| `recovery_01` | `HelbrosMarnieGeneralRecoveryV4_extracted` | 799201 | `artifacts/team_rocket_recovery/n10_epoch4_confirmation/panel/recovery_01.jsonl` |
| `recovery_02` | `HelbrosMarnieGeneralRecoveryV4_extracted` | 799202 | `artifacts/team_rocket_recovery/n10_epoch4_confirmation/panel/recovery_02.jsonl` |

The four collectors may execute concurrently. Only row counts and process
health may be monitored before completion. Win counts and arm comparisons stay
unread until all four manifests close.

After collection, the frozen analyzer must receive the four files in table
order and write exactly
`artifacts/team_rocket_recovery/n10_epoch4_confirmation/panel/analysis.json`
with 50,000 bootstrap replicates and analysis seed 799001. No partial analysis,
extension, substituted seed, or alternate checkpoint is allowed.

