# N7 Development Panel Execution Lock

Status: frozen after runtime parity and before the first N7 development-panel game.

This document instantiates the already preregistered 2,400-game panel in `N7_ARENA_MARNIE_SECOND_AWR_PROTOCOL.md`. It changes no arm, opponent, seed, game count, seat, gate, or tie-break rule.

## Frozen integrity chain

- N7 protocol SHA-256: `f5aa03037ec3ed15d4347a95aa45cb1112b080ebd50f411092c144f110399365`
- training recovery amendment SHA-256: `854472c4baf31d293e6ff2c20decfc68eaad8e40b6e86211a241580ba89b410d`
- recovery manifest SHA-256: `6bc493471fcbb6a83e0ce8ed871d338df864330c0888760edb9b7d318c05a292`
- arms manifest SHA-256: `adaa00256988097da0d53ee8371e17b1e442529caa47c533357553eb679d795b`
- runtime-parity report SHA-256: `41206832f72ac150619a9c9f2676e555a95a463b784468cf1391c130c2f0a9a1`
- runtime-parity result: 2,456/2,456 official-engine acted decisions matched, zero mismatches, 40 games
- exact live candidate bundle tree SHA-256: `11988ad1e5ab0b81d993ad54bf32b1a903caf89b6a4040aacf57e0f1d6a8480a`
- exact deck SHA-256: `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`

Arm order is fixed as `control`, `epoch_01`, `epoch_02`, `epoch_03`, `epoch_04`. Their model hashes are fixed in the arms manifest.

## Frozen execution code

- arm builder: `scripts/build_n7_awr_arms.py`, SHA-256 `265bda1fd41e3f542182e1e7c87ebb82fcdf52626f0efe7b9b15998d6224be82`
- runtime verifier: `scripts/verify_n7_awr_runtime_parity.py`, SHA-256 `73f732ae8b84a8e3febe7efe7bc87b54fc0c8b9dc9623d2d0f70c895d8a6e0a6`
- panel collector: `scripts/collect_n7_awr_panel.py`, SHA-256 `00da34276630e3d6a126256c9a6f8343cbc57d53e77a105753db98452c716a84`
- panel analyzer: `scripts/analyze_n7_awr_panel.py`, SHA-256 `90d8158b0c1098951d8b8885c89eff7e31aa70aa5cef6a48e216c9db674b5a82`
- official evaluator: `ptcg_agent_lab/tools/evaluate.py`, SHA-256 `d676c7a385b73140d60cbce1eac65ee024c0dee8c7c6746e902e4388e67f5303`

All actors use public observations only. Rocket is always seat 1. PyTorch uses one thread. The collector is strict no-fallback execution and records official-engine provenance in every shard manifest.

## Frozen opponents and shards

- `AshMarnieGeneralV110_extracted` bundle tree SHA-256: `3bb5e40bb1583e31470eee263ed615c9207427c7b5d2199e80fbe3571e2cbb2b`
- `HelbrosMarnieGeneralRecoveryV4_extracted` bundle tree SHA-256: `caf361fd297ab63b24338650b6e396a296b961bb6ba61d2519799a90dff738f4`

Each shard contains 120 games per arm, 600 total:

| Output | Opponent | Schedule seed |
|---|---|---:|
| `development_panel/v110_01.jsonl` | AshMarnieGeneralV110 | 795101 |
| `development_panel/v110_02.jsonl` | AshMarnieGeneralV110 | 795102 |
| `development_panel/recovery_01.jsonl` | HelbrosMarnieGeneralRecoveryV4 | 795201 |
| `development_panel/recovery_02.jsonl` | HelbrosMarnieGeneralRecoveryV4 | 795202 |

The four shards total 2,400 games: 480 per arm and 240 per arm/opponent. Shards may execute concurrently because each uses a separate process, official-engine instance, output file, and opponent worker. Concurrency does not change schedules or random seeds.

## Frozen analysis

Analyze exactly the four complete shards with 20,000 bootstrap replicates and seed `794001`. An arm passes only with:

- stratified aggregate delta at least +0.020;
- both opponent deltas strictly above 0;
- aggregate bootstrap 95% lower endpoint strictly above -0.005;
- 480 valid second-seat games and zero exception fallbacks.

If multiple arms pass, choose the largest aggregate delta; an exact tie chooses the earlier epoch. No partial-shard inspection may change this rule. The analysis has no deployment authority.
