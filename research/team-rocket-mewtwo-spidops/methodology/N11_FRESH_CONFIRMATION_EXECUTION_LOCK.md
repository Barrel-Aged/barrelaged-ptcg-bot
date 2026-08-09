# N11 Fresh Confirmation Execution Lock

Status: **frozen before collection**  
Protocol: `rocket-n11-fresh-epoch4-confirmation/v1`  
Frozen: 2026-08-01

## Immutable implementation

- Protocol document SHA-256:
  `99b291b49ef17da3413c41113fd642defda0f68f526ebd68845309f4c9d46f17`.
- Collector SHA-256:
  `d2335db8639f5736aeb1618a63f004811593a359248b9c9eab631c88e880e597`.
- Analyzer SHA-256:
  `a65d910bcb41970717c213ac4d262aee379c8f516da16a1e8ca8053d9d0eb93f`.
- Frozen collector dependency SHA-256:
  `99882f0cdc4e722a03f45b1bf54bf2cf71e963ab0c1395b5e41b429b91f9a589`.
- Frozen analyzer dependency SHA-256:
  `db11727af97580a7c67a06118fc7cbc3e223faa95bf7b1718701f16523fc94da`.
- Two-arm manifest SHA-256:
  `d60ab7cbd53a49cc9a924f8a9b8d7445937f7db4b9c80d90ef7a04c007bdf443`.
- Control model SHA-256:
  `3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b`.
- Candidate model SHA-256:
  `bdafc1ce36f7e572554930a1a9b1beff541a351ece09f934d0416b4ce269a020`.
- Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`.

Both N11 scripts compile under the same Python 3.12 runtime used by the official
panel collectors. Existing exact packaged-runtime parity remains 850/850
combined decisions across control and candidate, with zero mismatches.

## Frozen collection

Each shard contains 400 games per arm and 800 rows:

| Shard | Opponent | Seed | Output |
|---|---|---:|---|
| `v110_01` | `AshMarnieGeneralV110_extracted` | 800101 | `artifacts/team_rocket_recovery/n11_fresh_epoch4_confirmation/panel/v110_01.jsonl` |
| `v110_02` | `AshMarnieGeneralV110_extracted` | 800102 | `artifacts/team_rocket_recovery/n11_fresh_epoch4_confirmation/panel/v110_02.jsonl` |
| `recovery_01` | `HelbrosMarnieGeneralRecoveryV4_extracted` | 800201 | `artifacts/team_rocket_recovery/n11_fresh_epoch4_confirmation/panel/recovery_01.jsonl` |
| `recovery_02` | `HelbrosMarnieGeneralRecoveryV4_extracted` | 800202 | `artifacts/team_rocket_recovery/n11_fresh_epoch4_confirmation/panel/recovery_02.jsonl` |

All four may run concurrently. Only row counts, process health, and manifest
existence may be observed before completion. Any collector failure or host
interruption rejects N11; no shard may be restarted.

After four complete manifests, run the frozen analyzer on the four files in
table order and write exactly
`artifacts/team_rocket_recovery/n11_fresh_epoch4_confirmation/panel/analysis.json`.
The analyzer must use 50,000 bootstrap replicates and seed 800001. No alternate
analysis, sample extension, seed substitution, or N10 data is allowed.

