# N15 Development Panel Execution Lock

Status: **frozen before development match collection**  
Protocol: `rocket-n15-prospective-terminal-consolidation/v1`

## Immutable evidence and implementation

- N15 protocol SHA-256:
  `f6da7eb8f3bf6d8f64e9d5b3670c0cadee4fe171f4b7525f1569f07b1af32beb`;
- N15 terminal model SHA-256:
  `354a41f5d4146ae2b48b3935653a0e47f535150eebcd37e90aa866d98d0c5ee7`;
- exact Team Rocket deck SHA-256:
  `256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852`;
- arm manifest SHA-256:
  `e46197852d7e88d069b54d9ecc5cc978b5d0d5b0c731cb34409a9d88c1fa34ef`;
- candidate bundle tree SHA-256:
  `982fa183a4ed963bda201c5dd01b55cf316e37848797441ef34d4848e0c7ff74`;
- packaged runtime parity SHA-256:
  `95f6d3f74883baf73395a2504772ac014607659bf5a9fe0a9060b37b57cf8ca8`;
- collector SHA-256:
  `e90626b37c892301dc4d283afa8335c05a96c8c7610ce46a0e1c41cbf608d7bb`;
- analyzer SHA-256:
  `e338856f1231825282c82888347472a284414cb5eae855dad9f1aed9e716d051`;
- official-game helper SHA-256:
  `f3cb21219efbf82c94a1bebe1d90b712fcd02cb28037397a02176d2ce5047f6b`;
- opponent-worker helper SHA-256:
  `a8eb4808fd29109ebd851b9802b72245b3353b0d64f97384a1aebd1948b80b85`;
- bootstrap helper SHA-256:
  `cabf1026ab21429ec11fb37b8e98760be3ef354a3303316c9f09d72340281df8`.

The packaged-path audit matched 1,049/1,049 official-engine Rocket decisions
across control and candidate with zero mismatches. Both frozen opponent bundle
trees and their deck/model/main hashes are embedded in the collector. The four
collector/analyzer scripts compile, focused tests pass, and the panel directory
did not exist when this lock was written.

## Locked collection

Each command collects 120 games per arm, 240 rows, with Rocket always seat 1:

```powershell
python scripts/collect_n15_development_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 815101 --output artifacts/team_rocket_recovery/n15_terminal_consolidation/development_panel/v110_01.jsonl
python scripts/collect_n15_development_panel.py --opponent work/arena_bundles/AshMarnieGeneralV110_extracted --seed 815102 --output artifacts/team_rocket_recovery/n15_terminal_consolidation/development_panel/v110_02.jsonl
python scripts/collect_n15_development_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 815201 --output artifacts/team_rocket_recovery/n15_terminal_consolidation/development_panel/recovery_01.jsonl
python scripts/collect_n15_development_panel.py --opponent work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted --seed 815202 --output artifacts/team_rocket_recovery/n15_terminal_consolidation/development_panel/recovery_02.jsonl
```

The four commands may run concurrently. No row, win count, or partial manifest
may be inspected or analyzed until all four 240-row shards and all four sidecar
manifests are complete. Process health, completed-row counters printed by the
collectors, file existence, and final row counts may be monitored.

If collection is interrupted, preserve every partial unopened. Do not restart
or substitute a shard without a separate recovery lock.

## Locked analysis

After all four shards seal, run exactly:

```powershell
python scripts/analyze_n15_development_panel.py
```

The analyzer uses seed `815002` and 20,000 bootstrap replicates. Advancement
requires strict correctness, at least +2.0 percentage points equally
stratified, a strictly positive point estimate against both opponents, and a
95% bootstrap lower bound strictly greater than -0.5 points.

A pass authorizes only a new, fresh confirmation protocol. A failure ends N15.
Neither result authorizes deployment. The live bot remains unchanged.
